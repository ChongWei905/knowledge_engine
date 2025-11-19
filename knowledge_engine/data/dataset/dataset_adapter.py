from abc import ABC, abstractmethod
from typing import Any, Iterable, Dict


class DatasetAdapter(ABC):
    """
    Abstract adapter for datasets that unifies underlying data structures across execution modes.

    Role:
    - Provides a consistent interface for `UnifiedDataset` regardless of the native container
      (e.g., `list[Document]`, Ray `Dataset`).

    Contract:
    - count(obj): Return the number of items.
    - iter_rows(obj): Iterate native row records (e.g., dict with serialized document bytes).
    - iter_docs(obj): Iterate deserialized `Document` objects.
    - map_batches(obj, fn, **kwargs): Apply a batch mapping function and return the native container.
    """

    @abstractmethod
    def count(self, obj: Any) -> int: ...

    @abstractmethod
    def iter_rows(self, obj: Any) -> Iterable[Dict[str, Any]]: ...

    @abstractmethod
    def iter_docs(self, obj: Any) -> Iterable[Any]: ...

    @abstractmethod
    def map_batches(self, obj: Any, fn, **kwargs) -> Any: ...