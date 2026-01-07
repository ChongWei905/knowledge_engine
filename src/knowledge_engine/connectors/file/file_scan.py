import logging
import mimetypes
from functools import partial
from typing import Union, Optional, Callable, cast, Any

from pyarrow._fs import FileSystem, FileSelector
from ray.data.datasource import FileMetadataProvider

from knowledge_engine.data.dataset import UnifiedDataset
from knowledge_engine.data.docid import mkdocid
from knowledge_engine.data.document import Document
from knowledge_engine.transforms.base.scan import Scan
from knowledge_engine.utils.ray_utils import RayPathParser

logger = logging.getLogger(__name__)

class FileScan(Scan):
    """A base scan class for file based data"""

    def __init__(
        self,
        paths: Union[str, list[str]],
        *,
        filesystem: Optional[FileSystem] = None,
        override_num_blocks: Optional[int] = None,
        **resource_args,
    ):
        """
        Initialize a file scan node.

        Args:
            paths: Input path string or list; supports local paths or URIs (e.g., s3://).
                A single string is normalized to a list.
            filesystem: Optional PyArrow FileSystem; if not provided, the filesystem may be inferred to match paths.
            override_num_blocks: Overrides the number of input partitions, affecting read parallelism and
                split granularity.
            resource_args: Resource and scheduling hints (in Ray mode may include `compute`, etc.).
        """
        super().__init__(**resource_args)
        assert len(paths) > 0
        if isinstance(paths, str):
            paths = [paths]
        assert isinstance(paths, list)
        self._paths = paths
        self._filesystem = filesystem
        if self._filesystem is None:
            self._try_infer_fs()

        self.override_num_blocks = override_num_blocks

    def _is_s3_scheme(self) -> bool:
        if isinstance(self._paths, str):
            return self._paths.startswith("s3:")
        else:
            return all(path.startswith("s3:") for path in self._paths)

    def _try_infer_fs(self):
        # todo: try to infer the file system from uri
        pass
        # from sycamore.utils.pyarrow import infer_fs
        #
        # common_fs = None
        # new_paths = []
        # for p in self._paths:
        #     (fs, root) = infer_fs(p)
        #     new_paths.append(root)
        #     if common_fs is None:
        #         common_fs = fs
        #     if not isinstance(fs, common_fs.__class__):
        #         logger.warning(
        #             f"Different paths infer multiple filesystems. {self._paths[0]}"
        #             + f"  gives {common_fs.__class__.__name__} and {p} gives"
        #             + f" {fs.__class__.__name__}.  Using no fs and hoping."
        #         )
        #         return
        #
        # assert common_fs is not None
        # self._filesystem = common_fs
        # self._paths = new_paths

class BinaryScan(FileScan):
    """Scan data file into raw bytes

        For each file, BinaryScan creates one Document in the form of
        {"doc_id": nanoid,
         "content": {"binary": xxx, "text": None},
          "properties": {"path": xxx}, "filetype": yyy}.

        Note: if you specify filter_paths_by_extension = False, you need to make sure
        all the files that are scanned can be processed by the pipeline. Many pipelines
        include file-type specific steps.
        """

    def __init__(
        self,
        paths: Union[str, list[str]],
        *,
        binary_format: str,
        override_num_blocks: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        filter_paths_by_extension: bool = True,
        concurrency: int | None = None,
        **resource_args,
    ):
        """
        Initialize a binary file scan node that filters by extension and produces `Document` objects.

        Args:
            paths: Input path(s), supports local paths and URIs.
            binary_format: Binary type identifier (e.g., "pdf"); used to set document type and extension filtering.
            override_num_blocks: Overrides partition count, affecting read parallelism and splitting.
            filesystem: Optional PyArrow FileSystem.
            metadata_provider: File-level metadata provider to enrich document properties.
            filter_paths_by_extension: Whether to filter input files automatically by extension.
            resource_args: Resource and scheduling hints.

        Output:
            - Acts as a leaf node in the plan; supports reading and producing datasets in local or Ray execution modes.
        """
        super().__init__(
            paths,
            override_num_blocks=override_num_blocks,
            filesystem=filesystem,
            **resource_args,
        )
        self._binary_format = binary_format.lower() if binary_format is not None else None
        self._metadata_provider = metadata_provider
        self._filter_paths_by_extension = filter_paths_by_extension
        self._path_filter = None
        self.concurrency = concurrency

    def execute_ray(self, **kwargs) -> "UnifiedDataset":
        file_extensions = [self.format()] if self._filter_paths_by_extension else None

        from ray.data import read_binary_files
        from ray.data.datasource import PathPartitionFilter, PathPartitionParser

        # TODO: Consider refactoring to use kwargs = self._get_read_args() for better extensibility
        # when adding new read arguments in the future
        partition_filter: Optional[Callable[[dict[str, str]], bool]] = None
        if self._path_filter is not None:
            partition_filter = PathPartitionFilter(
                cast(PathPartitionParser, RayPathParser()), partial(self._path_filter, read_binary=True)
            )
        shuffle = None if partition_filter is None else "files"

        try:
            files = read_binary_files(
                self._paths,
                include_paths=True,
                filesystem=self._filesystem,
                override_num_blocks=self.override_num_blocks,
                ray_remote_args=self.resource_args,
                file_extensions=file_extensions,
                partition_filter=partition_filter,
                shuffle=shuffle,
                concurrency=self.concurrency
            )
        except ValueError as e:

            from ray.data import from_items

            if self._path_filter is not None and "No input files found to read." in str(e):
                return from_items(items=[])
            raise

        ds = files.map(self._to_document_ray, **self.resource_args)
        return UnifiedDataset.from_ray(ds)


    def execute_local(self, **kwargs) -> "UnifiedDataset":
        if isinstance(self._paths, str):
            paths = [self._paths]
        else:
            paths = self._paths

        docs = []

        for orig_path in paths:
            from knowledge_engine.utils.pyarrow.fs import cross_check_infer_fs

            (filesystem, path) = cross_check_infer_fs(self._filesystem, orig_path)
            if self._filesystem is None:
                self._filesystem = filesystem

            path_info = filesystem.get_file_info(path)
            if path_info.is_file:
                bytes = self._read_bytes_local(path_info)
                row = self._to_document(bytes, path_info.path)
                docs.append(row)
            else:
                for info in filesystem.get_file_info(FileSelector(path, recursive=True)):
                    bytes = self._read_bytes_local(info)
                    row = self._to_document(bytes, info.path)
                    docs.append(row)

        return UnifiedDataset.from_local(docs)

    def _read_bytes_local(self, info) -> bytes:
        if not info.is_file:
            return bytes()
        if self._filter_paths_by_extension and not info.path.lower().endswith(self.format()):
            return bytes()
        if self._path_filter is not None and not self._path_filter(info.path, True):
            return bytes()

        assert self._filesystem
        with self._filesystem.open_input_file(info.path) as file:
            return file.read()

    def _to_document(self, binary_representation: bytes, path: str) -> Document:
        document = Document()

        document.doc_id = mkdocid("f")
        document.type = self._binary_format
        document.binary_representation = binary_representation

        if self._is_s3_scheme():
            path = "s3://" + path

        document.properties.update({"path": path})
        if "filetype" not in document.properties and self._binary_format is not None:
            document.properties["filetype"] = self._file_mime_type()
        if self._metadata_provider:
            document.properties.update(self._metadata_provider.get_metadata(dict["path"]))
        # todo: make docid with a stable path
        # if self._path_filter is not None:
        #     from sycamore.materialize_config import MRRNameGroup
        #     MRRNameGroup.make_docid(document)
        return document

    def _to_document_dict(self, binary_representation: bytes, path: str) -> dict[str, bytes]:
        document = self._to_document(binary_representation, path)
        return {"doc": document.serialize()}

    def _file_mime_type(self):
        # binary_format is an extension, make it into a filename.
        (ftype, encoding) = mimetypes.guess_type("foo." + self._binary_format)
        if ftype is not None:
            return ftype
        ret = f"application/{self._binary_format}"
        logger.warning(f"Unrecognized extenstion {self._binary_format}; using {ret}")
        return ret

    def _to_document_ray(self, row: dict[str, Any]) -> dict[str, bytes]:
        return self._to_document_dict(row["bytes"], row["path"])

    def format(self):
        return self._binary_format


