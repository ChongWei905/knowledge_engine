from typing import Optional, TYPE_CHECKING

from knowledge_engine.context import Context
from knowledge_engine.data.document import MetadataDocument, Document
from knowledge_engine.transforms.extraction.extractor import Extractor
from knowledge_engine.transforms import Node

if TYPE_CHECKING:
    from knowledge_engine.data.docset.docset_writer import DocSetWriter


class DocSet:
    """A DocSet, short for “Document Set”, is a lazy pipeline that can be processed in a distributed
    manner. It starts with a read step, and is followed by a series of transformation on the
    DocSet.

    Provides a variety of transformations on DocSets to help customers modify unstructured
    data easily. Also provides a variety of readers and writers to start and finish a
    pipeline.

    Usage:
    - Planning: chain transform builders to produce a new `DocSet` without executing.
      - extract: extract contents and save in elements
      - explode: explode elements to separate documents
    - Execution:
      - take_all: materialize results as a list of `Document`
      - write: materialize to sinks
      - execute: run for side effects.

    Notes:
    - Lazily evaluated: no work is performed until an execution method is called.
    - Supports metadata documents; `take_all(include_metadata=True)` returns them, otherwise they
      are filtered out.
    """

    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan


    # ======================================
    # planning methods
    # ======================================

    def extract(
        self, extractor: Extractor, **kwargs
    ) -> "DocSet":
        from knowledge_engine.transforms.extraction import Extraction

        plan = Extraction(self.plan, extractor=extractor, **kwargs)
        return DocSet(self.context, plan)

    def explode(self, **resource_args) -> "DocSet":
        from knowledge_engine.transforms.base.explode import Explode

        explode = Explode(self.plan, **resource_args)
        return DocSet(self.context, explode)


    # ======================================
    # execution methods
    # ======================================

    def take_all(self, limit: Optional[int] = None, include_metadata: bool = False, **kwargs) -> list[Document]:
        from knowledge_engine import Execution

        docs = []
        for doc in Execution(self.context).execute_iter(self.plan, **kwargs):
            if include_metadata or not isinstance(doc, MetadataDocument):
                docs.append(doc)

            if limit is not None and len(docs) > limit:
                raise ValueError(f"docset exceeded limit of {limit} docs")

        return docs

    @property
    def write(self) -> "DocSetWriter":
        from knowledge_engine.data.docset.docset_writer import DocSetWriter

        return DocSetWriter(self.context, self.plan)

    def execute(self, **kwargs) -> None:
        from knowledge_engine import Execution

        # todo: materialize read reliability
        for doc in Execution(self.context).execute_iter(self.plan, **kwargs):
            pass
