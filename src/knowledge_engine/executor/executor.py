from typing import Iterable, Dict

from knowledge_engine.exec_mode import ExecMode
from knowledge_engine.context import Context
from knowledge_engine.data.document import Document
from knowledge_engine.transforms import Node
from knowledge_engine.executor.engines import Engine
from knowledge_engine.executor.engines import LocalEngine
from knowledge_engine.executor.engines import RayEngine

_engines: Dict[ExecMode, Engine] = {
    ExecMode.LOCAL: LocalEngine(),
    ExecMode.RAY: RayEngine()
}


class Execution:
    """
    Orchestrates plan execution by selecting an engine based on the Context's ExecMode.
    """

    def __init__(self, context: Context):
        self._context = context
        self._exec_mode = context.exec_mode
        self._engine = _engines[self._exec_mode]

    def execute_iter(self, plan: Node, **kwargs) -> Iterable[Document]:
        plan = self._apply_rules(plan)
        self._prepare(plan)
        ds = self._engine.execute_plan(plan, self._context, **kwargs)
        for row in ds.iter_docs():
            yield row
        print(ds.native().stats())

        plan.traverse(visit=lambda n: n.finalize())


    def _apply_rules(self, plan: Node) -> Node:
        # todo: optimizing plan executing
        return plan

    def _prepare(self, plan: Node):
        # todo: add prepare operations to a queue
        return plan