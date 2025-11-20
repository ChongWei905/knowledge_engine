from dataclasses import dataclass, field
from typing import Optional, Any

from knowledge_engine.exec_mode import ExecMode


@dataclass
class Context:
    """
    A class to implement a knowledge_engine Context, which initializes a Ray Worker and provides the ability
    to read data into a DocSet
    """

    exec_mode: ExecMode = ExecMode.LOCAL
    ray_args: Optional[dict[str, Any]] = None

    params: dict[str, Any] = field(default_factory=dict)

    @property
    def read(self):
        from knowledge_engine.data.docset import DocSetReader
        return DocSetReader(self)

def init(exec_mode=ExecMode.RAY, ray_args: Optional[dict[str, Any]] = None, **kwargs) -> Context:
    """
    Initialize a new Context.
    """
    if ray_args is None:
        ray_args = {}

    # Set Logger for driver only, we consider worker_process_setup_hook
    # or runtime_env/config file for worker application log
    from knowledge_engine.utils import logger

    logger.setup_logger()

    return Context(exec_mode=exec_mode, ray_args=ray_args, **kwargs)
