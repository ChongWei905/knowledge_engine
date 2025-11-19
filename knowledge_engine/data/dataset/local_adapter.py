from typing import Any, Iterable, Dict

from pandas import DataFrame

from knowledge_engine.data.dataset.dataset_adapter import DatasetAdapter
from knowledge_engine.data.document import Document


class LocalDatasetAdapter(DatasetAdapter):
    """Local Dataset Adapter which implements DatasetAdapter interface.

    Note that the native dataset is a list of `Document`.
    """
    def count(self, obj: list[Document]) -> int:
        return len(obj)

    def iter_rows(self, obj: list[Document]) -> Iterable[Dict[str, Any]]:
        for row in obj:
            yield {'doc': row.serialize()}

    def iter_docs(self, obj: list[Document]) -> Iterable[Document]:
        return obj

    def map_batches(self, obj: list[Document], fn, **kwargs) -> DataFrame:
        # todo: unchecked map
        return fn(obj)



