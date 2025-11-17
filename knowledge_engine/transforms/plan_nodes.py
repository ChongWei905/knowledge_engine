from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable

from knowledge_engine.data.dataset.unified_dataset import UnifiedDataset

class NodeTraverseOrder(Enum):
    BEFORE = 1
    AFTER = 2

class NodeTraverse:
    """NodeTraverse allows for complicated traversals

    For simple use cases, call node.traverse({before,visit,after}=fn)

    - before is called before traversing children.
    - after is called after traversing children.
    - visit is called over each node in an unspecified order, and is easier to use since the
      function returns nothing.
    - once is called one time at the very start, and enables multi-pass transforms.
    """

    BEFORE = NodeTraverseOrder.BEFORE
    AFTER = NodeTraverseOrder.AFTER

    def __init__(
        self,
        before: Optional[Callable[["Node"], "Node"]] = None,
        visit: Optional[Callable[["Node"], None]] = None,
        after: Optional[Callable[["Node"], "Node"]] = None,
    ):
        self.before_fn = before
        self.visit_fn = visit
        self.after_fn = after

    def once(self, context: "Context", node: "Node") -> "Node":
        # Called one time at the start of rewriting on the root of the tree.
        # Enables multi-pass traversals
        return node

    # Called before traversing children
    def before(self, node: "Node") -> "Node":
        if self.before_fn is None:
            return node
        return self.before_fn(node)

    # Called before traversing children, convenience function for single node mutating operations
    def visit(self, node: "Node") -> None:
        if self.visit_fn is not None:
            self.visit_fn(node)

    # Called after traversing children
    def after(self, node: "Node") -> "Node":
        if self.after_fn is None:
            return node
        return self.after_fn(node)

class Node(ABC):

    def __init__(
        self,
        children: list[Optional["Node"]],
        parallelism: Optional[int] = None,
        **resource_args
    ):
        self.children = children
        self.parallelism = parallelism
        self.resource_args = resource_args

    def __str__(self):
        return "node"

    @abstractmethod
    def execute_ray(self, **kwargs) -> UnifiedDataset:
        pass

    @abstractmethod
    def execute_local(self, **kwargs) -> UnifiedDataset:
        pass

    def prepare(self) -> Optional[Callable]:
        pass

    def finalize(self) -> None:
        pass

    def traverse_down(self, f: Callable[["Node"], "Node"]) -> "Node":
        """
        Allows a function to be applied to a node first and then all of its children
        """
        f(self)
        self.children = [c.traverse_down(f) for c in self.children if c is not None]
        return self

    def traverse_up(self, f: Callable[["Node"], "Node"]) -> "Node":
        """
        Allows a function to be applied to all of a node's children first and then itelf
        """
        self.children = [c.traverse_up(f) for c in self.children if c is not None]
        f(self)
        return self

    def traverse(
        self,
        obj: Optional[NodeTraverse] = None,
        before: Optional[Callable[["Node"], "Node"]] = None,
        visit: Optional[Callable[["Node"], None]] = None,
        after: Optional[Callable[["Node"], "Node"]] = None,
    ) -> "Node":
        """
        Traverse the node tree, functions will be converted to an object.
        See NodeTraverse for the semantics.
        """
        if obj is None:
            assert before is not None or visit is not None or after is not None
            obj = NodeTraverse(before=before, visit=visit, after=after)
        else:
            assert before is None and visit is None and after is None

        return self._traverse(obj)

    def _traverse(self, obj: NodeTraverse) -> "Node":
        n = obj.before(self)
        obj.visit(self)
        n.children = [c._traverse(obj) for c in n.children if c is not None]
        return obj.after(n)


class LeafNode(Node):
    def __init__(self, **resource_args):
        super().__init__([], **resource_args)

    def __str__(self, **resource_args):
        return "leaf"

class UnaryNode(Node):
    def __init__(self, child: Optional[Node], **resource_args):
        super().__init__([child], **resource_args)

    def __str__(self):
        return "unary"

    def child(self) -> Node:
        assert self.children[0] is not None
        return self.children[0]
