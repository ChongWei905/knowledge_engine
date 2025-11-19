import posixpath
from typing import Optional, Callable

from pyarrow._fs import FileSystem

from knowledge_engine.data.dataset import UnifiedDataset
from knowledge_engine.data.document import Document
from knowledge_engine.transforms import Node
from knowledge_engine.transforms.base.write import Write
from knowledge_engine.utils.file_utils import default_filename, default_doc_to_bytes


class FileWriter(Write):
    """Write implementation of file connector that writes out binary or text representation.

    Each document is written to a separate file.
    """

    def __init__(
        self,
        plan: Node,
        path: str,
        filesystem: Optional[FileSystem] = None,
        filename_fn: Callable[[Document], str] = default_filename,
        doc_to_bytes_fn: Callable[[Document], bytes] = default_doc_to_bytes,
        include_metadata: bool = False,
        **ray_remote_args,
    ):
        """Initializes a FileWriter instance.

        Args:
            plan: A plan representing the DocSet to write out.
            path: The path prefix to write to. Should include the scheme.
            filesystem: The pyarrow.fs FileSystem to use.
            filename_fn: A function for generating a file name. Takes a Document
                and returns a unique name that will be appended to path.
            doc_to_bytes_fn: A function from a Document to bytes for generating the data to write.
                Defaults to using text_representation if available, or binary_representation
                if not.
            include_metadata: Whether to include metadata documents in writing processes.
            ray_remote_args: Arguments to pass to the underlying execution environment.
        """

        super().__init__(plan, **ray_remote_args)
        self.path = path
        self.filesystem = filesystem
        self.filename_fn = filename_fn
        self.doc_to_bytes_fn = doc_to_bytes_fn
        self.include_metadata = include_metadata
        self.ray_remote_args = ray_remote_args

    def execute_ray(self, **kwargs) -> "UnifiedDataset":
        from knowledge_engine.connectors.file.file_writer_ray import _FileDataSink

        dataset = self.child().execute_ray()

        dataset.native().write_datasink(
            _FileDataSink(
                self.path,
                filesystem=self.filesystem,
                filename_fn=self.filename_fn,
                doc_to_bytes_fn=self.doc_to_bytes_fn,
            ),
            ray_remote_args=self.ray_remote_args,
        )

        return dataset

    def execute_local(self) -> "UnifiedDataset":
        from knowledge_engine.utils.pyarrow.fs import cross_check_infer_fs
        from knowledge_engine.data.document import MetadataDocument

        (filesystem, path) = cross_check_infer_fs(self.filesystem, self.path)

        ds = self.child().execute_local()
        all_docs = ds.native()

        for d in all_docs:
            if isinstance(d, MetadataDocument) and not self.include_metadata:
                continue
            bytes = self.doc_to_bytes_fn(d)
            file_path = posixpath.join(path, self.filename_fn(d))
            with filesystem.open_output_stream(str(file_path)) as file:
                file.write(bytes)

        return ds