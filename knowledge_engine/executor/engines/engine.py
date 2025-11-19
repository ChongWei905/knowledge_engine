from abc import ABC, abstractmethod
from typing import Callable

from knowledge_engine.context import Context
from knowledge_engine.data.dataset import UnifiedDataset
from knowledge_engine.data.dataset.dataset_adapter import DatasetAdapter
from knowledge_engine.transforms import Node


class Engine(ABC):

    @abstractmethod
    def get_dataset_adapter(self) -> "DatasetAdapter":
        raise NotImplementedError

    @abstractmethod
    def get_execute_func(self) -> Callable:
        raise NotImplementedError

    @abstractmethod
    def execute_plan(self, plan: Node, context: Context, **kwargs) -> "UnifiedDataset":
        raise NotImplementedError
