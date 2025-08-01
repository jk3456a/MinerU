from typing import List, Union
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
from PIL import Image


class YOLOv8MFDModel:
    def __init__(
        self,
        weight: str,
        device: str = "cpu",
        imgsz: int = 1888,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        # GPU优化点：YOLO v8公式检测模型加载到GPU
        # 建议：使用export()导出为TensorRT格式可获得2-5倍加速
        self.model = YOLO(weight).to(device)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def _run_predict(
        self,
        inputs: Union[np.ndarray, Image.Image, List],
        is_batch: bool = False
    ) -> List:
        # GPU优化点：YOLO推理，自动处理批量输入
        # 建议：使用stream=True参数可以在批量推理时节省内存
        preds = self.model.predict(
            inputs,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device
        )
        # GPU优化点：将结果从GPU转移到CPU
        # 建议：如果后续处理也在GPU上，可以延迟转移
        return [pred.cpu() for pred in preds] if is_batch else preds[0].cpu()

    def predict(self, image: Union[np.ndarray, Image.Image]):
        return self._run_predict(image)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4
    ) -> List:
        results = []
        # GPU优化点：批量公式检测
        # 建议：根据图片分辨率和GPU显存动态调整batch_size
        with tqdm(total=len(images), desc="MFD Predict") as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx: idx + batch_size]
                batch_preds = self._run_predict(batch, is_batch=True)
                results.extend(batch_preds)
                pbar.update(len(batch))
        return results