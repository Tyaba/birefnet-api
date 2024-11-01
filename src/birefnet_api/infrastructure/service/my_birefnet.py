import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from birefnet_api.domain.service.segmenter import (
    SegmenterInterface,
    SegmentInput,
    SegmentOutput,
)
from birefnet_api.settings import Settings
from birefnet_api.utils.image import resize_image_keep_aspect
from birefnet_api.utils.logger import get_logger
from models.birefnet import BiRefNet

logger = get_logger(__name__)

settings = Settings()


class MyBiRefNet(SegmenterInterface):
    def __init__(self) -> None:
        self.birefnet: BiRefNet = load_birefnet()
        logger.info("birefnet loaded")

    def segment(self, segment_input: SegmentInput) -> SegmentOutput:
        # Data settings
        images: list[Image.Image]
        bboxes: list[tuple[int, int, int, int]]
        images, bboxes = zip(
            *[preprocess_image(image) for image in segment_input.images],
            strict=False,
        )
        image_size = (1024, 1024)
        transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # 全ての画像を処理するように変更
        input_images = torch.stack(
            [transform_image(image).to(settings.device) for image in images]
        )

        # Prediction
        with torch.no_grad():
            preds = self.birefnet(input_images)[-1].sigmoid().cpu()

        # 各画像に対してマスクを適用
        masks: list[list[list[int]]] = []
        segmented_images: list[Image.Image] = []
        for pred, image, bbox in zip(preds, images, bboxes, strict=False):
            pred = pred.squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image.size)
            image.putalpha(mask)
            cropped_mask = mask.crop(bbox)
            cropped_image = image.crop(bbox)
            masks.append(np.array(cropped_mask, dtype=np.uint8).tolist())
            segmented_images.append(cropped_image)
        return SegmentOutput(
            masks=masks,
            segment_images=segmented_images,
        )


def load_birefnet() -> BiRefNet:
    birefnet = load_from_local()
    if birefnet is None:
        birefnet = load_from_huggingface()
    birefnet = init_birefnet(birefnet)
    return birefnet


def load_from_local() -> BiRefNet | None:
    if not settings.birefnet_local_path.exists():
        return None
    logger.info(f"{settings.birefnet_local_path} exists. load from local")
    return BiRefNet.from_pretrained(settings.birefnet_local_path)


def load_from_huggingface() -> BiRefNet:
    logger.info(f"load from huggingface: {settings.birefnet_model_id}")
    return BiRefNet.from_pretrained(settings.birefnet_model_id)


def init_birefnet(birefnet: BiRefNet) -> BiRefNet:
    torch.set_float32_matmul_precision(["high", "highest"][0])
    birefnet.to(settings.device)
    birefnet.eval()
    return birefnet


def preprocess_image(
    image: Image.Image,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    # imageを1024x1024のキャンバスにpasteしてbboxを返す
    resized_image = resize_image_keep_aspect(image, 1024)
    bbox = (0, 0, resized_image.width, resized_image.height)
    pasted_image = Image.new("RGB", (1024, 1024))
    pasted_image.paste(resized_image, bbox)
    return pasted_image, bbox
