from abc import ABC, abstractmethod
from typing import Any, Iterable, Dict


class DatasetAdapter(ABC):
    @abstractmethod
    def count(self, obj: Any) -> int: ...

    @abstractmethod
    def iter_rows(self, obj: Any) -> Iterable[Dict[str, Any]]: ...

    @abstractmethod
    def iter_docs(self, obj: Any) -> Iterable[Any]: ...

    @abstractmethod
    def map_batches(self, obj: Any, fn, **kwargs) -> Any: ...