from fastapi import APIRouter

from birefnet_api.app.di import DI
from birefnet_api.domain.model.segment import SegmentInput, SegmentOutput
from birefnet_api.domain.service.segmenter import SegmenterInterface
from birefnet_api.usecase.segment import SegmentUsecase
from birefnet_api.user_interface.segmenter import (
    SegmentRequest,
    SegmentResponse,
)
from birefnet_api.utils.image import base642pil

router = APIRouter()
injector = DI()
usecase = injector.resolve(SegmentUsecase)
segmenter = injector.resolve(SegmenterInterface)


@router.post("/segment")
def segment(
    request: SegmentRequest,
) -> SegmentResponse:
    segment_input = SegmentInput(
        images=[base642pil(image_base64=image) for image in request.images]
    )
    segment_output: SegmentOutput = usecase.segment(images=segment_input.images)
    return SegmentResponse(masks=segment_output.masks)
