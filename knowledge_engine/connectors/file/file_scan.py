import logging
import mimetypes
from functools import partial
from typing import Union, Optional, Callable, cast, Any

from pandas import DataFrame
from pyarrow._fs import FileSystem, FileSelector
from ray.data.datasource import FileMetadataProvider

from knowledge_engine.data.dataset import UnifiedDataset
from knowledge_engine.data.docid import mkdocid
from knowledge_engine.data.document import Document
from knowledge_engine.transforms.base.scan import Scan
from knowledge_engine.utils.ray_utils import RayPathParser

logger = logging.getLogger(__name__)

class FileScan(Scan):

    def __init__(
        self,
        paths: Union[str, list[str]],
        *,
        filesystem: Optional[FileSystem] = None,
        parallelism: Optional[str] = None,
        override_num_blocks: Optional[int] = None,
        **resource_args,
    ):
        super().__init__(**resource_args)
        assert len(paths) > 0
        if isinstance(paths, str):
            paths = [paths]
        assert isinstance(paths, list)
        self._paths = paths
        self._filesystem = filesystem
        if self._filesystem is None:
            self._try_infer_fs()

        assert parallelism is None, "Use override_num_blocks; remove parameter after 2025-03-01"
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


    def __init__(
        self,
        paths: Union[str, list[str]],
        *,
        binary_format: str,
        parallelism: Optional[str] = None,
        override_num_blocks: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        filter_paths_by_extension: bool = True,
        **resource_args,
    ):
        super().__init__(
            paths,
            parallelism=parallelism,
            override_num_blocks=override_num_blocks,
            filesystem=filesystem,
            **resource_args,
        )
        self._binary_format = binary_format.lower() if binary_format is not None else None
        self._metadata_provider = metadata_provider
        self._filter_paths_by_extension = filter_paths_by_extension
        self._path_filter = None

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

        df = DataFrame()

        for orig_path in paths:
            from knowledge_engine.utils.pyarrow.fs import cross_check_infer_fs

            (filesystem, path) = cross_check_infer_fs(self._filesystem, orig_path)
            if self._filesystem is None:
                self._filesystem = filesystem

            path_info = filesystem.get_file_info(path)
            if path_info.is_file:
                bytes = self._read_bytes_local(path_info)
                row = self._to_document(bytes, path_info.path)
                df = df._append(row, ignore_index=True)
            else:
                for info in filesystem.get_file_info(FileSelector(path, recursive=True)):
                    bytes = self._read_bytes_local(info)
                    row = self._to_document(bytes, info.path)
                    df = df._append(row, ignore_index=True)

        return UnifiedDataset.from_local(df)

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

    def _to_document(self, binary_representation: bytes, path: str) -> dict[str, bytes]:
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
        return self._to_document(row["bytes"], row["path"])

    def format(self):
        return self._binary_format


