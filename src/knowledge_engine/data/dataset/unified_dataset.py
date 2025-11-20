from typing import Any, Dict, Iterable

from datasets import Dataset

from knowledge_engine.data.document import Document
from knowledge_engine.exec_mode import ExecMode
from knowledge_engine.data.dataset.dataset_adapter import DatasetAdapter
from knowledge_engine.data.dataset.local_adapter import LocalDatasetAdapter
from knowledge_engine.data.dataset.ray_adapter import RayDatasetAdapter

_adapters: Dict[ExecMode, DatasetAdapter] = {
    ExecMode.LOCAL: LocalDatasetAdapter(),
    ExecMode.RAY: RayDatasetAdapter()
}

class UnifiedDataset:
    """
    Unified view over datasets across execution modes (LOCAL, RAY).

    Wraps the native container and delegates operations to a mode-specific
    DatasetAdapter. Exposes a common surface: `count`, `iter_rows`, `iter_docs`,
    and `map_batches`, plus factory constructors `from_ray` and `from_local`.

    Please make sure to add corresponding adapter implementations if other modes
    are going to be supported.
    """

    def __init__(self, exec_mode: ExecMode, obj: Any):
        """
        Args:
            exec_mode: Execution mode determining which adapter to use.
            obj: Native dataset container (Ray `Dataset` for RAY, `list[Document]` for LOCAL).
        """
        self._exec_mode = exec_mode
        self._obj = obj
        self._adapter = _adapters[exec_mode]

    # ======================================
    # Accessors for metadata and underlying objects
    # ======================================

    @property
    def exec_mode(self) -> ExecMode:
        return self._exec_mode
    
    def native(self) -> Any:
        return self._obj

    # ======================================
    # Operations for data manipulation and retrieval
    # ======================================

    def count(self) -> int:
        return self._adapter.count(self._obj)

    def iter_rows(self) -> Iterable[Dict[str, Any]]:
        return self._adapter.iter_rows(self._obj)

    def iter_docs(self) -> Iterable[Document]:
        return self._adapter.iter_docs(self._obj)

    def map_batches(self, fn, **kwargs) -> "UnifiedDataset":
        obj = self._adapter.map_batches(self._obj, fn, **kwargs)
        return UnifiedDataset(self._exec_mode, obj)

    # ======================================
    # Factory methods for creating instances
    # ======================================

    @staticmethod
    def from_ray(ds: Dataset) -> "UnifiedDataset":
        return UnifiedDataset(ExecMode.RAY, ds)

    @staticmethod
    def from_local(docs: list[Document]) -> "UnifiedDataset":
        return UnifiedDataset(ExecMode.LOCAL, docs)

