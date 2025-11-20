import pytest

def test_ray_adapter_basic():
    try:
        import ray
        from ray.data import from_items
    except Exception:
        pytest.skip("ray unavailable")
    from knowledge_engine.data.dataset.ray_adapter import RayDatasetAdapter
    from knowledge_engine.data.document import Document
    docs = [Document(), Document()]
    docs[0].text_representation = "u"
    docs[1].text_representation = "v"
    rows = [d.to_row() for d in docs]
    ds = from_items(rows)
    adapter = RayDatasetAdapter()
    assert adapter.count(ds) == 2
    out_docs = list(adapter.iter_docs(ds))
    assert len(out_docs) == 2
    ds2 = adapter.map_batches(ds, lambda batch: batch)
    assert ds2.count() == 2