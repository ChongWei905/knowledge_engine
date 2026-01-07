

import os
from typing import Optional

from knowledge_engine.data.docid import mkdocid
from knowledge_engine.data.document import Document


def default_filename(doc: Document, auto_infer: bool = True, extension: Optional[str] = None) -> str:
    """Returns a default filename based on document_id and extension.

    If the doc_id is not set, a new uuid is generated.

    Args:
        doc: A sycamore.data.Document instance.
        extension: An optional extension that will be appended to the name following a '.'.
    """
    if doc.doc_id is None:
        base_name = mkdocid()
    else:
        base_name = str(doc.doc_id)

    if auto_infer:
        base_name = add_origin_name(base_name, doc)
        extension = infer_extension(extension, doc)

    if extension is not None and len(extension) > 0:
        return f"{base_name}.{extension.lstrip('.')}"

    return base_name

def add_origin_name(base_name: str, doc: Document) -> str:
    properties = doc.properties
    if properties is not None:
        if "path" in properties.keys():
            origin_name = properties.get("path", "")
            base_name = base_name + "-" + os.path.splitext(os.path.basename(origin_name))[0]
        if "parent_path" in properties.keys():
            origin_name = properties.get("parent_path", "")
            base_name = base_name + "-" + os.path.splitext(os.path.basename(origin_name))[0]
    type_str = doc.type
    if type_str in ("html graph", "json graph"):
        base_name = base_name + "-" + "knowledge_graph"
    if type_str == "json matched":
        base_name = base_name + "-" + "matched_knowledge"

    return base_name


def infer_extension(extension: str, doc: Document) -> str:
    if extension is not None and len(extension) > 0:
        return extension
    type_str = doc.type
    if type_str in ("json matched", "json graph"):
        return "json"
    if type_str == "html graph":
        return "html"
    return ""


def default_doc_to_bytes(doc: Document) -> bytes:
    """Returns the text_representation of the document if available or the binary representation if not.

    Args:
        doc: A sycamore.data.Document instance.
    """
    if doc.text_representation is not None:
        return doc.text_representation.encode("utf-8")
    elif doc.binary_representation is not None:
        return doc.binary_representation
    else:
        raise RuntimeError(f"No default content representation for Document {doc}")