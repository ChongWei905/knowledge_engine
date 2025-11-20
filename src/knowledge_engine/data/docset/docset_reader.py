from typing import Optional, Union

from pyarrow._fs import FileSystem
from ray.data.datasource import FileMetadataProvider

from knowledge_engine.context import Context
from knowledge_engine.connectors.file.file_scan import BinaryScan
from knowledge_engine.data.docset.docset import DocSet
from knowledge_engine.transforms import Node


class DocSetReader:
    """Builder for read operations that create a `DocSet` pipeline."""

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
        """
        Build a binary file scan node and return a `DocSet` rooted at that node.

        Args:
            paths: Input path string or list; supports local paths and URIs (e.g., s3://...).
            binary_format: File type identifier (e.g., "pdf"); used for document type and extension filtering.
            parallelism: Deprecated; prefer `override_num_blocks` to control input partitioning.
            override_num_blocks: Overrides number of input partitions; affects read parallelism.
            filesystem: Optional PyArrow `FileSystem` for explicit FS selection.
            metadata_provider: Optional provider to enrich per-file document properties.
            kwargs: Additional resource or execution hints propagated to the scan node.

        Returns:
            A `DocSet` whose plan is a `BinaryScan` configured with the provided parameters.
        """

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




