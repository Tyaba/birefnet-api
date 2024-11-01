from abc import ABCMeta, abstractmethod

from birefnet_api.domain.model.segment import SegmentInput, SegmentOutput


class SegmenterInterface(metaclass=ABCMeta):
    @abstractmethod
    def segment(self, segment_input: SegmentInput) -> SegmentOutput:
        pass
