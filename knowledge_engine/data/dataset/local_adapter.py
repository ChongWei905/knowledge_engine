from typing import Any, Iterable, Dict

from pandas import DataFrame

from knowledge_engine.data.dataset.dataset_adapter import DatasetAdapter
from knowledge_engine.data.document import Document


class LocalDatasetAdapter(DatasetAdapter):
    def count(self, obj: Any) -> int:
        return len(obj)

    def iter_rows(self, obj: Any) -> Iterable[Dict[str, Any]]:
        for row in obj:
            yield {'doc': row.serialize()}

    def iter_docs(self, obj: Any) -> Iterable[Document]:
        return obj

    def map_batches(self, obj: Any, fn, **kwargs) -> DataFrame:
        # todo: unchecked map
        return fn(obj)



