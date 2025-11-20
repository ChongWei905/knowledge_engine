import pytest
from knowledge_engine.data.dataset.unified_dataset import UnifiedDataset
from knowledge_engine.data.document import Document

def test_unified_dataset_local():
    docs = [Document(), Document()]
    docs[0].text_representation = "x"
    docs[1].text_representation = "y"
    ds = UnifiedDataset.from_local(docs)
    assert ds.count() == 2
    rows = list(ds.iter_rows())
    assert len(rows) == 2
    out_docs = list(ds.iter_docs())
    assert len(out_docs) == 2
    def fn(batch):
        return batch
    ds2 = ds.map_batches(fn)
    assert ds2.count() == 2

def test_unified_dataset_ray():
    try:
        import ray
        from ray.data import from_items
        from knowledge_engine.data.dataset.unified_dataset import UnifiedDataset
        from knowledge_engine.data.document import Document
    except Exception:
        pytest.skip("ray or unified_dataset unavailable")
    docs = [Document(), Document()]
    docs[0].text_representation = "m"
    docs[1].text_representation = "n"
    rows = [d.to_row() for d in docs]
    ds_native = from_items(rows)
    ds = UnifiedDataset.from_ray(ds_native)
    assert ds.count() == 2
    out_docs = list(ds.iter_docs())
    assert len(out_docs) == 2
    ds2 = ds.map_batches(lambda batch: batch)
    assert ds2.count() == 2