import abc
from typing import Sequence

class BaseDescriptorCalculator(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        pass
    
    @abc.abstractmethod
    def calculate(self, molecule, **kwargs) -> Sequence[float|int]:
        pass
    
    @property
    @abc.abstractmethod
    def descriptor_names(self) -> Sequence[str]:
        return self.descriptor_summaries

    @property
    def descriptor_summaries(self) -> Sequence[str]:
        pass
    
    @property
    def descriptor_count(self) -> int:
        return len(self.descriptor_names)
