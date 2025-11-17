from typing import Optional, Any, Callable

from knowledge_engine.data.document import Document
from knowledge_engine.transforms.base.base_map_transform import BaseMapTransform, get_name_from_callable
from knowledge_engine.transforms import Node


class Map(BaseMapTransform):
    """
    Map is a transformation class for applying a callable function to each document in a dataset.

    If f is a class type, constructor_args and constructor_kwargs can be used to provide arguments when
    initializing the class

    Use args, kwargs to pass additional args to the function call. The following 2 are equivalent:

    # option 1:
    docset.map(lambda f_wrapped: f(*my_args, **my_kwargs))

    # option 2:
    docset.map(f, args=my_args, kwargs=my_kwargs)

    If f is a class type, when using ray execution, the class will be mapped to an agent that
    will be instantiated a fixed number of times. By default that will be once, but you can
    change that with:
        .. code-block:: python

           ctx.map(ExampleClass, parallelism=num_instances)

    Example:
         .. code-block:: python

            def custom_mapping_function(document: Document) -> Document:
                # Custom logic to transform the document
                return transformed_document

            map_transformer = Map(input_dataset_node, f=custom_mapping_function)
            transformed_dataset = map_transformer.execute()
    """

    def __init__(self, child: Optional[Node], *, f: Any, **kwargs):
        super().__init__(child, f=Map.wrap(f), **{"name": get_name_from_callable(f), **kwargs})

    @staticmethod
    def wrap(f: Any) -> Callable[[list[Document]], list[Document]]:
        if isinstance(f, type):
            # mypy doesn't understand the dynamic class inheritence.
            class _Wrap(f):  # type: ignore[valid-type,misc]
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def __call__(self, docs, *args, **kwargs):
                    assert isinstance(docs, list)
                    for d in docs:
                        assert isinstance(d, Document)
                    s = super()
                    return [s.__call__(d, *args, **kwargs) for d in docs]

            return _Wrap
        else:

            def _wrap(docs, *args, **kwargs):
                assert isinstance(docs, list)
                for d in docs:
                    assert isinstance(d, Document)
                return [f(d, *args, **kwargs) for d in docs]

            return _wrap

    # todo: false run
    # def run(self, d: Document) -> Document:
    #     ret = self._local_process([d])
    #     assert len(ret) == 1
    #     return ret[0]