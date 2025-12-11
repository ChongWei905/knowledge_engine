import logging
from typing import Any

from knowledge_engine.data.document import Document
from knowledge_engine.llms.prompts import PromptProcessor, RenderedPrompt, RenderedMessage

logger = logging.getLogger(__name__)


class DocumentTextProcessor(PromptProcessor):
    def __init__(self, tokenize: bool = False, add_generation_prompt: bool = True):
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt

    def render_document(self, doc: Document, **kwargs) -> RenderedPrompt:
        text =f"what does it mean: {doc.elements[0].text_representation[:10]}"
        processor: Any = kwargs.get("processor", None)
        if not processor:
            prompt_text = text
        else:
            message = [{"role": "user", "content": text}]
            try:
                prompt_text = processor.apply_chat_template(
                    message, tokenize=self.tokenize, add_generation_prompt=self.add_generation_prompt)
            except ValueError as e:
                logger.warning(f"Got error {e} trying to apply chat template. Using raw prompt instead.")
                prompt_text = text
        messages = [RenderedMessage(
            role="user",
            content=prompt_text,
            images=None
        )]
        return RenderedPrompt(messages=messages)


