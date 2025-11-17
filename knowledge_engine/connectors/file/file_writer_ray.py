from typing import Any, Callable, Iterable, Optional

import posixpath
from pyarrow.fs import FileSystem, FileType
from ray.data.block import Block, BlockAccessor
from ray.data.datasource import Datasink
from ray.data.datasource.path_util import _resolve_paths_and_filesystem
from ray.data._internal.execution.interfaces import TaskContext
from urllib.parse import urlparse

from knowledge_engine.data.document import Document, MetadataDocument
from knowledge_engine.utils.file_utils import default_filename, default_doc_to_bytes


class _FileDataSink(Datasink):
    def __init__(
        self,
        path: str,
        filesystem: Optional[FileSystem] = None,
        filename_fn: Callable[[Document], str] = default_filename,
        doc_to_bytes_fn: Callable[[Document], bytes] = default_doc_to_bytes,
        makedirs: bool = True,
        include_metadata: bool = False,
    ):
        (paths, self._filesystem) = _resolve_paths_and_filesystem(path, filesystem)
        self._root = paths[0]
        if self._root == "":
            self._root = "./"
        self._filename_fn = filename_fn
        self._doc_to_bytes_fn = doc_to_bytes_fn
        self._makedirs = makedirs
        self._include_metadata = include_metadata

    def on_write_start(self) -> None:
        if not self._makedirs:
            return

        # This follows Ray logic to skip attempting to
        # create "directories" for s3 filesystems.
        parsed_uri = urlparse(self._root)
        is_s3_uri = parsed_uri.scheme == "s3"

        if not is_s3_uri and self._filesystem.get_file_info(self._root).type is FileType.NotFound:
            self._filesystem.create_dir(self._root, recursive=True)

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> Any:
        for block in blocks:
            b = BlockAccessor.for_block(block).to_arrow().to_pylist()
            for _, row in enumerate(b):
                doc = Document.from_row(row)
                if isinstance(doc, MetadataDocument) and not self._include_metadata:
                    continue
                bytes = self._doc_to_bytes_fn(doc)
                path = posixpath.join(self._root, self._filename_fn(doc))
                with self._filesystem.open_output_stream(path) as file:
                    file.write(bytes)