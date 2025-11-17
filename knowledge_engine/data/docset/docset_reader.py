from typing import Optional, Union

from pyarrow._fs import FileSystem
from ray.data.datasource import FileMetadataProvider

from knowledge_engine.context import Context
from knowledge_engine.connectors.file.file_scan import BinaryScan
from knowledge_engine.data.docset.docset import DocSet
from knowledge_engine.transforms import Node


class DocSetReader:

    def __init__(self, context: Context, plan: Optional[Node] = None):
        self._context = context
        self._plan = plan

    def binary(
        self,
        paths: Union[str, list[str]],
        binary_format: str,
        parallelism: Optional[str] = None,
        override_num_blocks: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        **kwargs,
    ) -> DocSet:

        scan = BinaryScan(
            paths,
            binary_format=binary_format,
            parallelism=parallelism,
            override_num_blocks=override_num_blocks,
            filesystem=filesystem,
            metadata_provider=metadata_provider,
            **kwargs
        )
        return DocSet(self._context, scan)




