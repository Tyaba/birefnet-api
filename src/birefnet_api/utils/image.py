import base64
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image


def base642pil(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))
    return image


def pil2base64(
    image: Image.Image, format: Literal["PNG", "JPEG", "WEBP"] = "JPEG"
) -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        print(f"{image_dir} does not exist. Please check the path and try again.")
    image_suffixes = {".jpg", ".png", ".jpeg", ".gif"}
    if image_dir.suffix in image_suffixes:
        print(f"{image_dir} is a file, not a directory. Use the file as an image.")
        return [image_dir]
    image_paths = []
    for image_suffix in image_suffixes:
        # recursive search for images in child directories
        image_paths += list(image_dir.glob(f"**/*{image_suffix}"))
    image_paths = sorted(image_paths)
    print(f"Loaded {len(image_paths)} images from {image_dir}")
    return image_paths


def resize_image_keep_aspect(image: Image.Image, long_size: int) -> Image.Image:
    # 縦横比を維持しつつ、長辺がlong_sizeになるようにリサイズします。
    width, height = image.size

    if height > width:
        new_height = long_size
        new_width = int(new_height * width / height)
    else:
        new_width = long_size
        new_height = int(new_width * height / width)

    # 画像を新しい解像度にリサイズします。
    img_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img_resized
