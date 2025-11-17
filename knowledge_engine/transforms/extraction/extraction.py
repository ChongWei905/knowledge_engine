from typing import Optional, Callable

from knowledge_engine.data.document import Document
from knowledge_engine.data.element import Element
from knowledge_engine.transforms.base.base_map_transform import BaseMapTransform, get_name_from_callable
from knowledge_engine.transforms import Node
from knowledge_engine.transforms.base.map import Map
from knowledge_engine.transforms.extraction.extractor import Extractor


class Extraction(BaseMapTransform):

    def __init__(self, child: Optional[Node], *, extractor: Extractor, **kwargs):
        f = Extraction.wrap(extractor)
        super().__init__(child, f=f, **{"name": get_name_from_callable(f), **kwargs})

    @staticmethod
    def wrap(extractor: Extractor) -> Callable[[list[Document]], list[Document]]:
        def extract_doc(doc: Document) -> Document:
            pdf_bytes = doc.data["binary_representation"]
            text = extractor.extract(pdf_bytes=pdf_bytes)
            binary = text.encode("utf-8")

            element = Element()
            element.type = "Text"
            element.element_index = 0
            element.binary_representation = binary
            element.text_representation = text

            # todo: add other properties
            doc.elements.append(element)
            return doc

        return Map.wrap(extract_doc)














