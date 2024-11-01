from PIL import Image
from pydantic import BaseModel, ConfigDict


class SegmentInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    images: list[Image.Image]


class SegmentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    masks: list[list[list[int]]]
    segment_images: list[Image.Image]
