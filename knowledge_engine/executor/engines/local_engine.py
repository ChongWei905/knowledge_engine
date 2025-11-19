from typing import Callable

from knowledge_engine import Context
from knowledge_engine.data.dataset import UnifiedDataset
from knowledge_engine.data.dataset.dataset_adapter import DatasetAdapter
from knowledge_engine.transforms import Node
from knowledge_engine.executor.engines import Engine


class LocalEngine(Engine):

    def get_dataset_adapter(self) -> "DatasetAdapter":
        from knowledge_engine.data.dataset.local_adapter import LocalDatasetAdapter
        return LocalDatasetAdapter()

    def get_execute_func(self) -> Callable:
        def execute(n: Node) -> "UnifiedDataset":
            return n.execute_local()
        return execute

    def execute_plan(self, plan: Node, context: Context, **kwargs) -> "UnifiedDataset":
        return plan.execute_local(**kwargs)