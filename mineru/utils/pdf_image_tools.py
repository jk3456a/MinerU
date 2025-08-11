# Copyright (c) Opendatalab. All rights reserved.
from io import BytesIO

import pypdfium2 as pdfium
from loguru import logger
from PIL import Image

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.pdf_reader import image_to_b64str, image_to_bytes, page_to_image
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def _render_pages_chunk(args):
    """Worker: 在子进程中渲染一组页面，返回 (page_index, image_dict) 列表。

    注意：每个子进程各自打开 PdfDocument，避免跨进程共享对象带来的不安全。
    """
    pdf_bytes, dpi, indices = args
    results = []
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        for i in indices:
            try:
                page = doc[i]
                pil_img, scale = page_to_image(page, dpi=dpi)
                img_base64 = image_to_b64str(pil_img)
                image_dict = {"img_base64": img_base64, "img_pil": pil_img, "scale": scale}
                results.append((i, image_dict))
            except Exception:
                continue
        try:
            doc.close()
        except Exception:
            pass
    except Exception:
        pass
    return results
from .hash_utils import str_sha256


def pdf_page_to_image(page: pdfium.PdfPage, dpi=200) -> dict:
    """Convert pdfium.PdfDocument to image, Then convert the image to base64.

    Args:
        page (_type_): pdfium.PdfPage
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img_base64': str, 'img_pil': pil_img, 'scale': float }
    """
    # 固定使用传入的 dpi，不开放通过环境变量修改，以保持一致的渲染策略
    pil_img, scale = page_to_image(page, dpi=dpi)

    # 允许通过环境变量跳过页级 base64 生成，减少 CPU 与内存
    enable_b64 = str(os.getenv("MINERU_ENABLE_PAGE_BASE64", "0")).lower() in ["1", "true", "yes", "on"]
    img_base64 = image_to_b64str(pil_img) if enable_b64 else ""

    image_dict = {
        "img_base64": img_base64,
        "img_pil": pil_img,
        "scale": scale,
    }
    return image_dict


def load_images_from_pdf(
    pdf_bytes: bytes,
    dpi=200,
    start_page_id=0,
    end_page_id=None,
):
    images_list = []
    pdf_doc = pdfium.PdfDocument(pdf_bytes)
    pdf_page_num = len(pdf_doc)
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
    if end_page_id > pdf_page_num - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = pdf_page_num - 1

    # 生成目标页索引
    target_indices = [i for i in range(pdf_page_num) if start_page_id <= i <= end_page_id]

    # 并行渲染控制：优先线程池（MINERU_PDF_RENDER_THREADS），否则进程池（MINERU_PDF_RENDER_WORKERS），否则串行
    threads = 0
    workers = 0
    try:
        threads = int(os.getenv("MINERU_PDF_RENDER_THREADS", "0"))
    except Exception:
        threads = 0
    try:
        workers = int(os.getenv("MINERU_PDF_RENDER_WORKERS", "0"))
    except Exception:
        workers = 0

    total_pages = len(target_indices)

    if threads and threads > 1 and total_pages > 1:
        logger.info(f"PDF render mode: threads={threads}, pages={total_pages}")
        chunk_size = max(1, (total_pages + threads - 1) // threads)
        chunks = [target_indices[i:i + chunk_size] for i in range(0, total_pages, chunk_size)]

        futures = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for indices in chunks:
                futures.append(executor.submit(_render_pages_chunk, (pdf_bytes, dpi, indices)))
            tmp = []
            for fut in as_completed(futures):
                try:
                    tmp.extend(fut.result())
                except Exception:
                    continue
        tmp.sort(key=lambda x: x[0])
        images_list = [image_dict for _, image_dict in tmp]
    elif workers and workers > 1 and total_pages > 1:
        logger.info(f"PDF render mode: processes={workers}, pages={total_pages}")
        chunk_size = max(1, (total_pages + workers - 1) // workers)
        chunks = [target_indices[i:i + chunk_size] for i in range(0, total_pages, chunk_size)]

        futures = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for indices in chunks:
                futures.append(executor.submit(_render_pages_chunk, (pdf_bytes, dpi, indices)))
            tmp = []
            for fut in as_completed(futures):
                try:
                    tmp.extend(fut.result())
                except Exception:
                    continue
        tmp.sort(key=lambda x: x[0])
        images_list = [image_dict for _, image_dict in tmp]
    else:
        logger.info(f"PDF render mode: serial, pages={total_pages}")
        for index in target_indices:
            page = pdf_doc[index]
            image_dict = pdf_page_to_image(page, dpi=dpi)
            images_list.append(image_dict)

    return images_list, pdf_doc


def cut_image(bbox: tuple, page_num: int, page_pil_img, return_path, image_writer: FileBasedDataWriter, scale=2):
    """从第page_num页的page中，根据bbox进行裁剪出一张jpg图片，返回图片路径 save_path：需要同时支持s3和本地,
    图片存放在save_path下，文件名是:
    {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。"""

    # 拼接文件名
    filename = f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"

    # 老版本返回不带bucket的路径
    img_path = f"{return_path}_{filename}" if return_path is not None else None

    # 新版本生成平铺路径
    img_hash256_path = f"{str_sha256(img_path)}.jpg"
    # img_hash256_path = f'{img_path}.jpg'

    crop_img = get_crop_img(bbox, page_pil_img, scale=scale)

    img_bytes = image_to_bytes(crop_img, image_format="JPEG")

    image_writer.write(img_hash256_path, img_bytes)
    return img_hash256_path


def get_crop_img(bbox: tuple, pil_img, scale=2):
    scale_bbox = (
        int(bbox[0] * scale),
        int(bbox[1] * scale),
        int(bbox[2] * scale),
        int(bbox[3] * scale),
    )
    return pil_img.crop(scale_bbox)


def images_bytes_to_pdf_bytes(image_bytes):
    # 内存缓冲区
    pdf_buffer = BytesIO()

    # 载入并转换所有图像为 RGB 模式
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # 第一张图保存为 PDF，其余追加
    image.save(pdf_buffer, format="PDF", save_all=True)

    # 获取 PDF bytes 并重置指针（可选）
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes
