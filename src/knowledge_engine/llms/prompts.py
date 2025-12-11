from dataclasses import dataclass
from typing import Optional, List

from PIL import Image

from knowledge_engine.data.document import Document
from knowledge_engine.data.element import Element


@dataclass
class RenderedMessage:
    """Represents a message per the LLM messages interface - i.e. a role and a content string

        Args:
            role: the role of this message. e.g. for OpenAI should be one of "user", "system", "assistant"
            content: the content of this message
            images: optional list of images to include in this message.
        """

    role: str
    content: str
    tokenized_content: Optional[List[int]] = None
    images: Optional[list[Image.Image]] = None


@dataclass
class RenderedPrompt:
    messages: list[RenderedMessage]
    # more fields may be added later

    def to_human_readable(self) -> str:
        """For debugging purposes, render the prompt as a human readable string."""
        return "\n".join(
            f"------------------------\n{m.role}\n-----------------------\n{m.content}\n" for m in self.messages
        )

class PromptProcessor:

    def render_document(self, doc: Document, **kwargs) -> RenderedPrompt:
        """Render this prompt, given this document as context.
        Used in llm_map

        Args:
            doc: The document to use to populate the prompt

        Returns:
            A fully rendered prompt that can be sent to an LLM for inference
        """
        raise NotImplementedError(f"render_document is not implemented for {self.__class__.__name__}")

    def render_element(self, element: Element, doc: Document, **kwargs) -> RenderedPrompt:
        """Render this prompt, given this element and its parent document as context.
        Used in llm_map_elements

        Args:
            element: The element to use to populate the prompt
            doc: parent document of the element

        Returns:
            A fully rendered prompt that can be sent to an LLM for inference
        """
        raise NotImplementedError(f"render_element is not implemented for {self.__class__.__name__}")

    def render_multiple_documents(self, docs: list[Document], **kwargs) -> RenderedPrompt:
        """Render this prompt, given a list of documents as context.
        Used in llm_reduce

        Args:
            docs: The list of documents to use to populate the prompt

        Returns:
            A fully rendered prompt that can be sent to an LLM for inference"""
        raise NotImplementedError(f"render_multiple_documents is not implemented for {self.__class__.__name__}")

    def render_multiple_elements(self, elements: list[Element], doc: Document, **kwargs) -> RenderedPrompt:
        """Render this prompt, given a list of elements from a document as context.

        Args:
            elements: The list of elements to use to populate the prompt
            doc: The parent document of the elements

        Returns:
            A fully rendered prompt that can be sent to an LLM for inference
        """
        raise NotImplementedError(f"render_multiple_elements is not implemented for {self.__class__.__name__}")

