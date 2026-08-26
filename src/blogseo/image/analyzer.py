"""圖片前處理：讀檔、縮圖、轉成 LLM 可接受的格式。

在本地先把圖片縮到合理尺寸有兩個好處：省下上傳流量，以及大幅降低 vision
token 消耗。一篇十張圖的文章跑三個模型就是三十次 vision 呼叫，這裡省下來的
成本相當可觀。
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final

from PIL import Image, UnidentifiedImageError

from blogseo.config import MAX_IMAGE_BYTES, MAX_IMAGE_EDGE
from blogseo.errors import ImageNotFoundError, ImageProcessingError

#: Pillow 格式名稱對應到 LLM API 使用的 media type。
_FORMAT_MEDIA_TYPES: Final[dict[str, str]] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}

#: 各家 vision API 普遍接受的格式，其餘一律轉成 PNG。
_PASSTHROUGH_FORMATS: Final[frozenset[str]] = frozenset(_FORMAT_MEDIA_TYPES)

#: JPEG 重新編碼時嘗試的品質，由高到低。
_JPEG_QUALITIES: Final[tuple[int, ...]] = (85, 75, 65, 55)


@dataclass(frozen=True)
class PreparedImage:
    """已經處理好、可直接送進模型的圖片。

    Attributes:
        data: 圖片位元組。
        media_type: 對應的 MIME type。
        width: 送出時的寬度。
        height: 送出時的高度。
        original_width: 原始寬度。
        original_height: 原始高度。
        original_size_bytes: 原始檔案大小。
        resized: 是否曾被縮圖或重新編碼。
    """

    data: bytes
    media_type: str
    width: int
    height: int
    original_width: int
    original_height: int
    original_size_bytes: int
    resized: bool


def _flatten_for_jpeg(image: Image.Image) -> Image.Image:
    """把含透明度的圖片疊到白底上，以便存成 JPEG。

    Args:
        image: 來源圖片。

    Returns:
        RGB 模式的圖片。
    """
    if image.mode == "RGB":
        return image
    converted = image.convert("RGBA")
    background = Image.new("RGBA", converted.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, converted).convert("RGB")


def _encode(image: Image.Image, target_format: str) -> bytes:
    """把圖片編碼成位元組，必要時逐步降低 JPEG 品質以符合大小上限。

    Args:
        image: 來源圖片。
        target_format: Pillow 格式名稱。

    Returns:
        編碼後的位元組。

    Raises:
        ImageProcessingError: 編碼失敗。
    """
    try:
        if target_format == "JPEG":
            flattened = _flatten_for_jpeg(image)
            data = b""
            for quality in _JPEG_QUALITIES:
                buffer = BytesIO()
                flattened.save(buffer, format="JPEG", quality=quality, optimize=True)
                data = buffer.getvalue()
                if len(data) <= MAX_IMAGE_BYTES:
                    break
            return data

        buffer = BytesIO()
        save_image = image if image.mode in {"RGB", "RGBA", "L", "P"} else image.convert("RGBA")
        save_image.save(buffer, format=target_format, optimize=target_format == "PNG")
        data = buffer.getvalue()

        # PNG 壓不下來時改用 JPEG，總比整張送不出去好。
        if len(data) > MAX_IMAGE_BYTES and target_format == "PNG":
            return _encode(image, "JPEG")
        return data
    except OSError as exc:
        raise ImageProcessingError(f"圖片編碼失敗：{exc}") from exc


def prepare_image(path: Path, *, max_edge: int = MAX_IMAGE_EDGE) -> PreparedImage:
    """讀取圖片並轉成適合送進 LLM 的形式。

    尺寸與格式都符合條件時會直接沿用原始位元組，不做無謂的重新編碼。

    Args:
        path: 圖片路徑。
        max_edge: 長邊上限（px），超過就等比例縮小。

    Returns:
        處理好的圖片。

    Raises:
        ImageNotFoundError: 檔案不存在。
        ImageProcessingError: 圖片無法解碼或編碼。
    """
    if not path.is_file():
        raise ImageNotFoundError(f"找不到圖片：{path}")

    size_bytes = path.stat().st_size
    try:
        with Image.open(path) as image:
            original_format = (image.format or "").upper()
            original_width, original_height = image.size
            needs_resize = max(image.size) > max_edge
            unsupported = original_format not in _PASSTHROUGH_FORMATS
            too_large = size_bytes > MAX_IMAGE_BYTES
            animated = getattr(image, "n_frames", 1) > 1

            if not (needs_resize or unsupported or too_large or animated):
                with path.open("rb") as handle:
                    data = handle.read()
                return PreparedImage(
                    data=data,
                    media_type=_FORMAT_MEDIA_TYPES[original_format],
                    width=original_width,
                    height=original_height,
                    original_width=original_width,
                    original_height=original_height,
                    original_size_bytes=size_bytes,
                    resized=False,
                )

            working = image
            if animated:
                working.seek(0)
            working = working.copy()
            if needs_resize:
                working.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

            target_format = original_format if original_format in {"JPEG", "PNG"} else "PNG"
            data = _encode(working, target_format)
            width, height = working.size
    except UnidentifiedImageError as exc:
        raise ImageProcessingError(f"無法辨識的圖片格式：{path}") from exc
    except OSError as exc:
        raise ImageProcessingError(f"處理圖片失敗 {path}：{exc}") from exc

    media_type = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else _FORMAT_MEDIA_TYPES[target_format]
    return PreparedImage(
        data=data,
        media_type=media_type,
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
        original_size_bytes=size_bytes,
        resized=True,
    )
