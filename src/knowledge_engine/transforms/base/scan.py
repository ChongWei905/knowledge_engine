from abc import abstractmethod

from knowledge_engine.transforms.mixins import NonGPUUser, SingleThreadUser
from knowledge_engine.transforms.plan_nodes import LeafNode


class Scan(SingleThreadUser, NonGPUUser, LeafNode):
    def __init__(self, **resource_args):
        super().__init__(**resource_args)

    def __str__(self):
        return "scan"

    @abstractmethod
    def format(self):
        pass