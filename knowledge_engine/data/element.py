import json
from collections import UserDict
from typing import Optional, Any


class Element(UserDict):

    def __init__(self, element=None, /, **kwargs):
        super().__init__(element, **kwargs)
        if "properties" not in self.data:
            self.data["properties"] = {}

    @property
    def element_index(self) -> Optional[int]:
        """A unique identifier for the element within a Document. Represents an order within the document"""
        return self.data.get("properties", {}).get("_element_index")

    @element_index.setter
    def element_index(self, value: int) -> None:
        """Set the unique identifier of the element within a Document."""
        self.data["properties"]["_element_index"] = value

    @property
    def type(self) -> Optional[str]:
        return self.data.get("type")

    @type.setter
    def type(self, value: str) -> None:
        self.data["type"] = value

    @property
    def text_representation(self) -> Optional[str]:
        return self.data.get("text_representation")

    @text_representation.setter
    def text_representation(self, value: str) -> None:
        self.data["text_representation"] = value

    @property
    def binary_representation(self) -> Optional[bytes]:
        return self.data.get("binary_representation")

    @binary_representation.setter
    def binary_representation(self, value: bytes) -> None:
        self.data["binary_representation"] = value

    # @property
    # def bbox(self) -> Optional[BoundingBox]:
    #     return None if self.data.get("bbox") is None else BoundingBox(*self.data["bbox"])
    #
    # @bbox.setter
    # def bbox(self, bbox: BoundingBox) -> None:
    #     self.data["bbox"] = bbox.coordinates

    @property
    def properties(self) -> dict[str, Any]:
        return self.data.get("properties", None)

    @properties.setter
    def properties(self, properties: dict[str, Any]):
        self.data["properties"] = properties

    @properties.deleter
    def properties(self) -> None:
        self.data["properties"] = {}

    @property
    def embedding(self) -> Optional[list[float]]:
        """Get the embedding for this element."""
        return self.data.get("embedding")

    @embedding.setter
    def embedding(self, embedding: list[float]) -> None:
        """Set the embedding for this element."""
        self.data["embedding"] = embedding

    def __str__(self) -> str:
        """Return a pretty-printed string representing this Element."""
        d = {
            "type": self.type,
            "text_representation": self.text_representation[0:40] + "..." if self.text_representation else None,
            "binary_representation": (
                f"<{len(self.binary_representation)} bytes>" if self.binary_representation else None
            ),
            "embedding": (str(self.embedding[0:4]) + f"... <{len(self.embedding)} total>") if self.embedding else None,
            # todo: support bbox later
            # "bbox": str(self.bbox),
            "properties": {k: str(v) for k, v in self.properties.items()},
        }
        return json.dumps(d, indent=2)

def create_element(element_index: Optional[int] = None, **kwargs) -> Element:
    element: Element
    type = kwargs.get("type")
    if isinstance(type, str):
        type = type.lower()
    else:
        type = ""

    if type == "table":
        pass
        # todo: support table element later
        # if "properties" in kwargs:
        #     props = kwargs["properties"]
        #     kwargs["title"] = props.get("title")
        #     kwargs["columns"] = props.get("columns")
        #     kwargs["rows"] = props.get("rows")
        # if "table" in kwargs and isinstance(kwargs["table"], dict):
        #     table = Table.from_dict(kwargs["table"])
        #     kwargs["table"] = table
        #
        # element = TableElement(**kwargs)

    elif type in {"picture", "image", "figure"}:
        pass
        # todo: support image element later
        # if "properties" in kwargs:
        #     props = kwargs["properties"]
        #     kwargs["image_size"] = props.get("image_size")
        #     kwargs["image_mode"] = props.get("image_mode")
        #     kwargs["image_format"] = props.get("image_format")
        #
        # element = ImageElement(**kwargs)

    else:
        element = Element(**kwargs)
    if element_index is not None:
        element.element_index = element_index
    return element