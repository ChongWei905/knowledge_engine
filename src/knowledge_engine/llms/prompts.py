from dataclasses import dataclass
from typing import Optional, List

from PIL import Image


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
