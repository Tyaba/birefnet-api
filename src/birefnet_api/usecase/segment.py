"""python src/usecase/gsam.py \
    --image-path notebooks/images/abema_water.png \
    --text a 'product.'
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from PIL import Image

from birefnet_api.app.di import DI
from birefnet_api.domain.model.segment import SegmentInput, SegmentOutput
from birefnet_api.domain.service.segmenter import SegmenterInterface
from birefnet_api.utils.logger import get_logger

logger = get_logger(__name__)


class SegmentUsecase:
    def __init__(self, segmenter: SegmenterInterface) -> None:
        self.segmenter = segmenter

    def segment(
        self,
        images: list[Image.Image],
    ) -> SegmentOutput:
        logger.info(f"画像{len(images):,}枚からのsegmentationをします")
        segment_input = SegmentInput(images=images)
        segment_output = self.segmenter.segment(segment_input)
        return segment_output


def main(args: Namespace):
    injector = DI()
    segmenter = injector.resolve(SegmenterInterface)
    usecase = SegmentUsecase(segmenter=segmenter)
    image = Image.open(args.image_path)
    segment_output = usecase.segment(images=[image])
    print(segment_output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image-path",
        type=Path,
        help="path",
        required=True,
    )
    args = parser.parse_args()
    main(args=args)
