from typing import List, Dict, Union
from doclayout_yolo import YOLOv10
from tqdm import tqdm
import numpy as np
from PIL import Image


class DocLayoutYOLOModel:
    def __init__(
        self,
        weight: str,
        device: str = "cuda",
        imgsz: int = 1280,
        conf: float = 0.1,
        iou: float = 0.45,
    ):
        self.model = YOLOv10(weight).to(device)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def _parse_prediction(self, prediction) -> List[Dict]:
        layout_res = []

        # 容错处理
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return layout_res

        # 避免对同一结果多次 .cpu() 触发多个 D2H，将 boxes 一次性移到 CPU
        boxes_cpu = prediction.boxes.to('cpu') if hasattr(prediction.boxes, 'to') else prediction.boxes

        for xyxy, conf, cls in zip(
            boxes_cpu.xyxy,
            boxes_cpu.conf,
            boxes_cpu.cls,
        ):
            coords = list(map(int, xyxy.tolist()))
            xmin, ymin, xmax, ymax = coords
            layout_res.append({
                "category_id": int(cls.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 3),
            })
        return layout_res

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        prediction = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]
        return self._parse_prediction(prediction)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4
    ) -> List[List[Dict]]:
        results = []
        with tqdm(total=len(images), desc="Layout Predict") as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx: idx + batch_size]
                predictions = self.model.predict(
                    batch,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                )
                for pred in predictions:
                    results.append(self._parse_prediction(pred))
                pbar.update(len(batch))
        return results