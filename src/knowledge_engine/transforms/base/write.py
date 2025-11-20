from knowledge_engine.transforms.mixins import SingleThreadUser, NonGPUUser
from knowledge_engine.transforms.plan_nodes import UnaryNode, Node


class Write(SingleThreadUser, NonGPUUser, UnaryNode):
    def __init__(self, child: Node, **resource_args):
        super().__init__(child, **resource_args)

    def __str__(self):
        return "write"