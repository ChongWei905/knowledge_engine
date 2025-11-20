from typing import Optional, Callable

from knowledge_engine.data.document import Document
from knowledge_engine.transforms import Node
from knowledge_engine.transforms.base.base_map_transform import BaseMapTransform, get_name_from_callable


class FlatMap(BaseMapTransform):
    """
    FlatMap is a transformation class for applying a callable function to each document in a dataset and flattening
    the resulting list of documents.

    See :class:`Map` for additional arguments that can be specified and the option for the type of f.

    Example:
         .. code-block:: python

            def custom_flat_mapping_function(document: Document) -> list[Document]:
                # Custom logic to transform the document and return a list of documents
                return [transformed_document_1, transformed_document_2]

            flat_map_transformer = FlatMap(input_dataset_node, f=custom_flat_mapping_function)
            flattened_dataset = flat_map_transformer.execute()

    """

    def __init__(self, child: Optional[Node], *, f: Callable[[Document], list[Document]], **kwargs):
        super().__init__(child, f=FlatMap.wrap(f), **{"name": get_name_from_callable(f), **kwargs})

    @staticmethod
    def wrap(f: Callable[[Document], list[Document]]) -> Callable[[list[Document]], list[Document]]:
        if isinstance(f, type):

            class _Wrap(f):  # type: ignore[valid-type,misc]
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def __call__(self, docs, *args, **kwargs):
                    assert isinstance(docs, list)
                    s = super()
                    ret = []
                    for d in docs:
                        assert isinstance(d, Document)
                        ret.extend(s.__call__(d, *args, **kwargs))
                    return ret

            return _Wrap
        else:

            def _wrap(docs, *args, **kwargs):
                assert isinstance(docs, list)
                ret = []
                for d in docs:
                    assert isinstance(d, Document)
                    o = f(d, *args, **kwargs)
                    ret.extend(o)
                return ret

            return _wrap