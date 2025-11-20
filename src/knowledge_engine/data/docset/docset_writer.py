from typing import Optional, Callable

from pyarrow._fs import FileSystem

from knowledge_engine import Context, DocSet
from knowledge_engine.connectors.file.file_writer import FileWriter
from knowledge_engine.data.document import Document
from knowledge_engine.transforms import Node
from knowledge_engine.utils.file_utils import default_filename, default_doc_to_bytes


class DocSetWriter:
    """Materialization helper for `DocSet` pipelines."""

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
        """
        Write documents to a filesystem path using `FileWriter`.

        Parameters:
            path: Target path (local or URI) to write files under.
            filesystem: Optional PyArrow `FileSystem`; if None, inferred where possible.
            filename_fn: Function that derives output filenames from `Document`.
            doc_to_bytes_fn: Function that serializes a `Document` to bytes for writing.
            resource_args: Optional resource hints (e.g., Ray remote args) passed to the writer.

        Behavior:
            Builds a `FileWriter` node, wraps it into a `DocSet`, and executes to materialize outputs.
            Returns `None`; side effects are the written files.
        """

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