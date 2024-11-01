from pydantic import BaseModel, Field

from birefnet_api.usecase.segment import SegmentUsecase
from birefnet_api.utils.image import base642pil
from birefnet_api.utils.logger import get_logger

logger = get_logger(__name__)


class SegmentRequest(BaseModel):
    images: list[str]


class SegmentResponse(BaseModel):
    masks: list[list[list[int]]] = Field(
        default=...,
        title="segment mask",
        description="mask = masks[idx_image][x][y]のmask",
    )


class SegmentUserInterface:
    def __init__(self, use_case: SegmentUsecase) -> None:
        self.use_case = use_case

    def segment(self, request: SegmentRequest) -> SegmentResponse:
        logger.info("segment request received")
        images = [base642pil(image_base64=image) for image in request.images]
        segment_output = self.use_case.segment(images=images)
        logger.info("segment done")
        return SegmentResponse(masks=segment_output.masks)
