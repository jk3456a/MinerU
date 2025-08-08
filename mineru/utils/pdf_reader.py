# Copyright (c) Opendatalab. All rights reserved.
import base64
from io import BytesIO

from loguru import logger
from PIL import Image
from pypdfium2 import PdfBitmap, PdfDocument, PdfPage


def page_to_image(
    page: PdfPage,
    dpi: int = 144,  # changed from 200 to 144
    max_width_or_height: int = 2560,  # changed from 4500 to 2560
) -> tuple[Image.Image, float]:
    scale = dpi / 72

    long_side_length = max(*page.get_size())
    if (long_side_length*scale) > max_width_or_height:
        scale = max_width_or_height / long_side_length

    bitmap: PdfBitmap = page.render(scale=scale)  # type: ignore
    try:
        image = bitmap.to_pil()
    finally:
        try:
            bitmap.close()
        except Exception:
            pass
    return image, scale


def image_to_bytes(
    image: Image.Image,
    image_format: str = "PNG",  # 也可以用 "JPEG"
) -> bytes:
    with BytesIO() as image_buffer:
        image.save(image_buffer, format=image_format)
        return image_buffer.getvalue()


def image_to_b64str(
    image: Image.Image,
    image_format: str = "PNG",  # 也可以用 "JPEG"
) -> str:
    image_bytes = image_to_bytes(image, image_format)
    return base64.b64encode(image_bytes).decode("utf-8")


def pdf_to_images(
    pdf: str | bytes | PdfDocument,
    dpi: int = 144,
    max_width_or_height: int = 2560,
    start_page_id: int = 0,
    end_page_id: int | None = None,
) -> list[Image.Image]:
    doc_obj: PdfDocument = pdf if isinstance(pdf, PdfDocument) else PdfDocument(pdf)  # type: ignore[assignment]
    page_num = len(doc_obj)

    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else page_num - 1
    if end_page_id > page_num - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = page_num - 1

    images = []
    try:
        for i in range(start_page_id, end_page_id + 1):
            image, _ = page_to_image(doc_obj[i], dpi, max_width_or_height)
            images.append(image)
    finally:
        try:
            doc_obj.close()
        except Exception:
            pass
    return images


def pdf_to_images_bytes(
    pdf: str | bytes | PdfDocument,
    dpi: int = 144,
    max_width_or_height: int = 2560,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    image_format: str = "PNG",
) -> list[bytes]:
    images = pdf_to_images(pdf, dpi, max_width_or_height, start_page_id, end_page_id)
    return [image_to_bytes(image, image_format) for image in images]


def pdf_to_images_b64strs(
    pdf: str | bytes | PdfDocument,
    dpi: int = 144,
    max_width_or_height: int = 2560,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    image_format: str = "PNG",
) -> list[str]:
    images = pdf_to_images(pdf, dpi, max_width_or_height, start_page_id, end_page_id)
    return [image_to_b64str(image, image_format) for image in images]


# ----------------------------------------------------------------------------
# Streaming-friendly helpers
# ----------------------------------------------------------------------------
def load_images_from_pdf_iter(
    pdf: str | bytes | PdfDocument,
    dpi: int = 144,
    max_width_or_height: int = 2560,
    start_page_id: int = 0,
    end_page_id: int | None = None,
):
    """
    Incrementally render a PDF into PIL images page-by-page and yield as an iterator.

    This function is designed for streaming pipelines to reduce peak memory usage
    and improve time-to-first-result. It mirrors the behavior of `pdf_to_images`
    but avoids materializing the entire list in memory.

    Args:
        pdf: Path, raw bytes, or an existing PdfDocument
        dpi: Render DPI (default: 144)
        max_width_or_height: Max long side of rendered image (default: 2560)
        start_page_id: Start page index (inclusive)
        end_page_id: End page index (inclusive). None means last page

    Yields:
        dict with keys:
            - 'img_pil': PIL.Image.Image
            - 'scale': float (render scale used)

    Notes:
        - The underlying PdfDocument is closed automatically at the end.
        - This function intentionally yields dictionaries to match existing
          downstream expectations that use 'img_pil' and 'scale'.
    """
    doc_obj: PdfDocument = pdf if isinstance(pdf, PdfDocument) else PdfDocument(pdf)  # type: ignore[assignment]
    page_num = len(doc_obj)

    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else page_num - 1
    if end_page_id > page_num - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = page_num - 1

    try:
        for i in range(start_page_id, end_page_id + 1):
            image, scale = page_to_image(doc_obj[i], dpi, max_width_or_height)
            yield {"img_pil": image, "scale": scale}
    finally:
        try:
            doc_obj.close()
        except Exception:
            pass
