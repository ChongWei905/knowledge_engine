from typing import Optional, Callable

from knowledge_engine.data.document import Document
from knowledge_engine.data.element import Element
from knowledge_engine.transforms.base.base_map_transform import BaseMapTransform, get_name_from_callable
from knowledge_engine.transforms import Node
from knowledge_engine.transforms.base.map import Map
from knowledge_engine.transforms.extraction.extractor import Extractor


class Extraction(BaseMapTransform):
    """
    Map-style transform that applies an `Extractor` to each input `Document`, producing text and a new `Element`.
    """

    def __init__(self, child: Optional[Node], *, extractor: Extractor, **kwargs):
        f = Extraction.wrap(extractor)
        super().__init__(child, f=f, **{"name": get_name_from_callable(f), **kwargs})

    @staticmethod
    def wrap(extractor: Extractor) -> Callable[[list[Document]], list[Document]]:
        def extract_doc(doc: Document) -> Document:
            pdf_bytes = doc.data["binary_representation"]
            text = extractor.extract(pdf_bytes=pdf_bytes)
            # todo: currently just encodes with default encoding. As extractor extracts bytes to str, the
            #  encoding should be decided by the Extractor while extracting, and the encoding here should
            #  be identical with the encoding used in the extractor. In this case, it's inappropriate to let
            #  user decide the encoding here. However, it's confusing which encoding is actually used during
            #  pymupdf extraction.
            binary = text.encode()

            element = Element()
            element.type = "Text"
            element.element_index = 0
            element.binary_representation = binary
            element.text_representation = text

            # todo: add other properties
            doc.elements.append(element)
            return doc

        return Map.wrap(extract_doc)














