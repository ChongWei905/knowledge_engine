from typing import Any, Iterable, Dict

from ray.data import Dataset

from knowledge_engine.data.dataset.dataset_adapter import DatasetAdapter
from knowledge_engine.data.document import Document


class RayDatasetAdapter(DatasetAdapter):
    """Ray Dataset Adapter which implements DatasetAdapter interface.

    Note that the native dataset is a `ray.data.Dataset`.
    """
    def count(self, obj: Any) -> int:
        return obj.count()

    def iter_rows(self, obj: Any) -> Iterable[Dict[str, Any]]:
        return obj.iter_rows()

    def iter_docs(self, obj: Any) -> Iterable[Document]:
        for row in self.iter_rows(obj):
            yield Document.from_row(row)

    def map_batches(self, obj: Any, fn, **kwargs) -> Dataset:
        return obj.map_batches(fn, **kwargs)