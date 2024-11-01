from typing import TypeVar

from injector import Binder, Injector, singleton

from birefnet_api.domain.service.segmenter import SegmenterInterface
from birefnet_api.infrastructure.service.my_birefnet import MyBiRefNet

T = TypeVar("T")


@singleton
class DI:
    def __init__(self) -> None:
        self.injector = Injector(self._configure)

    def _configure(self, binder: Binder) -> None:
        binder.bind(SegmenterInterface, to=MyBiRefNet)

    def resolve(self, cls: type[T]) -> T:
        return self.injector.get(cls)
