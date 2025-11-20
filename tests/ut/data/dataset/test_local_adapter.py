import pytest
from knowledge_engine.data.dataset.local_adapter import LocalDatasetAdapter
from knowledge_engine.data.document import Document

def test_local_adapter_basic():
    docs = [Document(), Document()]
    docs[0].text_representation = "a"
    docs[1].text_representation = "b"
    adapter = LocalDatasetAdapter()
    assert adapter.count(docs) == 2
    rows = list(adapter.iter_rows(docs))
    assert len(rows) == 2
    assert "doc" in rows[0]
    out_docs = list(adapter.iter_docs(docs))
    assert len(out_docs) == 2
    def fn(batch):
        return batch
    out = adapter.map_batches(docs, fn)
    assert out == docs