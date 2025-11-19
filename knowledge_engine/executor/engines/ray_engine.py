import logging
from typing import Callable, Any

import knowledge_engine
from knowledge_engine import Context
from knowledge_engine.data.dataset import UnifiedDataset
from knowledge_engine.data.dataset.dataset_adapter import DatasetAdapter
from knowledge_engine.transforms import Node
from knowledge_engine.executor.engines import Engine


class RayEngine(Engine):
    """Ray Engine which implements Engine interface."""

    def get_dataset_adapter(self) -> "DatasetAdapter":
        from knowledge_engine.data.dataset.ray_adapter import RayDatasetAdapter
        return RayDatasetAdapter()

    def get_execute_func(self) -> Callable:
        def execute(n: Node) -> "UnifiedDataset":
            return n.execute_ray()
        return execute

    def execute_plan(self, plan: Node, context: Context, **kwargs) -> "UnifiedDataset":
        import ray

        if not ray.is_initialized():
            ray_args = context.ray_args
            self._ray_init(**ray_args)

        plan = plan.traverse(visit=self.visit_parallelism)
        return plan.execute_ray(**kwargs)

    @staticmethod
    def visit_parallelism(n: Node):
        assert isinstance(n, Node)
        if n.parallelism is None:
            n.resource_args.pop("compute", None)
        else:
            from ray.data import ActorPoolStrategy

            assert n.parallelism > 0
            n.resource_args["compute"] = ActorPoolStrategy(size=n.parallelism)

    @staticmethod
    def _ray_init(**ray_args: Any) -> None:
        import ray
        if ray.is_initialized():
            logging.warning("Ignoring explicit request to initialize ray when it is already initialized")
            return

        if "logging_level" not in ray_args:
            ray_args.update({"logging_level": logging.INFO})

        if "runtime_env" not in ray_args:
            new_val: dict[str, Any] = {"py_modules": [knowledge_engine]}  # Make mypy happy.
            ray_args["runtime_env"] = new_val

        if "worker_process_setup_hook" not in ray_args["runtime_env"]:
            # logging.error("Spurious log 0: If you do not see spurious log 1 & 2,
            # log messages are being dropped")
            ray_args["runtime_env"]["worker_process_setup_hook"] = RayEngine._ray_logging_setup

        ray.init(**ray_args)

    @staticmethod
    def _ray_logging_setup():
        # Some documentation for ray implies things should use the ray logger
        ray_logger = logging.getLogger("ray")
        ray_logger.setLevel(logging.INFO)
        # ray_logger.info("Spurious log 2: Verifying that log messages are propagated")

        ## Make the default logging show info messages
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Spurious log 1: Verifying that log messages are propagated")
        # logging.error("RayLoggingSetup-After-2Error")

        ## Verify that another logger would also log properly
        other_logger = logging.getLogger("other_logger")
        other_logger.setLevel(logging.INFO)
        # other_logger.info("RayLoggingSetup-After-3")