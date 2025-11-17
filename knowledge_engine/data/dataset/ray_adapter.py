from typing import Any, Iterable, Dict

from ray.data import Dataset

from knowledge_engine.data.dataset.dataset_adapter import DatasetAdapter

class RayDatasetAdapter(DatasetAdapter):
    def count(self, obj: Any) -> int:
        return obj.count()

    def iter_rows(self, obj: Any) -> Iterable[Dict[str, Any]]:
        return obj.iter_rows()

    def map_batches(self, obj: Any, fn, **kwargs) -> Dataset:
        return obj.map_batches(fn, **kwargs)