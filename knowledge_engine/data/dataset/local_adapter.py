from typing import Any, Iterable, Dict

from pandas import DataFrame

from knowledge_engine.data.dataset.dataset_adapter import DatasetAdapter

class LocalDatasetAdapter(DatasetAdapter):
    def count(self, obj: Any) -> int:
        return len(obj)

    def iter_rows(self, obj: Any) -> Iterable[Dict[str, Any]]:
        for row in obj.to_dict(orient="records"):
            yield row

    def map_batches(self, obj: Any, fn, **kwargs) -> DataFrame:
        # todo: unchecked map
        df = DataFrame(obj)
        results = []
        batch_size = kwargs.get("batch_size", 1000)
        for start_idx in range(0, len(df), batch_size):
            batch = df.iloc[start_idx:start_idx + batch_size]
            processed_batch = fn(batch)
            results.append(processed_batch)
        return df



