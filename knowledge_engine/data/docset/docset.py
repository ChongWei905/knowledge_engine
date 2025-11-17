from typing import Optional, TYPE_CHECKING

from knowledge_engine.context import Context
from knowledge_engine.data.document import MetadataDocument, Document
from knowledge_engine.transforms.extraction.extractor import Extractor
from knowledge_engine.transforms import Node

if TYPE_CHECKING:
    from knowledge_engine.data.docset.docset_writer import DocSetWriter


class DocSet:
    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan

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
    # executing methods
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
