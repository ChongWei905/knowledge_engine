from typing import Optional, Callable

from pyarrow._fs import FileSystem

from knowledge_engine import Context, DocSet
from knowledge_engine.connectors.file.file_writer import FileWriter
from knowledge_engine.data.document import Document
from knowledge_engine.transforms import Node
from knowledge_engine.utils.file_utils import default_filename, default_doc_to_bytes


class DocSetWriter:

    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan

    def files(
        self,
        path: str,
        filesystem: Optional[FileSystem] = None,
        filename_fn: Callable[[Document], str] = default_filename,
        doc_to_bytes_fn: Callable[[Document], bytes] = default_doc_to_bytes,
        **resource_args,
    ) -> None:

        file_writer: Node = FileWriter(
            self.plan,
            path,
            filesystem=filesystem,
            filename_fn=filename_fn,
            doc_to_bytes_fn=doc_to_bytes_fn,
            **resource_args,
        )
        ds = DocSet(self.context, file_writer)
        ds.execute()