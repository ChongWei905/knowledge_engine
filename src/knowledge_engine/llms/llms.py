import copy
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any

from PIL import Image

from knowledge_engine.llms.config import LLMModel, LLMMode
from knowledge_engine.llms.prompts import RenderedPrompt
from vllm import LLM


class LLMFactory(ABC):

    @abstractmethod
    def create(self) -> LLM:
        pass

    @abstractmethod
    def get_default_mode(self) -> LLMMode:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

class LLM(ABC):

    model: LLMModel

    def __init__(
        self,
        model_name: str,
        default_mode: LLMMode,
        default_llm_kwargs: Optional[dict[str, Any]] = None,
    ):
        self._model_name: str = model_name
        # currently don't support cache as it's not necessary for local llm?
        self._default_mode = default_mode
        self._default_llm_kwargs = default_llm_kwargs or {}

    def __str__(self):
        return f"{self.__class__.__name__}({self._model_name})"

    def default_mode(self) -> LLMMode:
        """Returns the default execution mode for the llm"""
        return self._default_mode

    def _merge_llm_kwargs(self, llm_kwargs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Merges the default LLM kwargs with any provided LLM kwargs.

        Prefers the passed in values if there is a conflict.
        """
        new_kwargs = copy.copy(self._default_llm_kwargs)
        new_kwargs.update(llm_kwargs or {})
        logging.debug(f"Merging LLM kwargs: {new_kwargs}")
        return new_kwargs

    @abstractmethod
    def generate(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        """Generates a response from the LLM for the given prompt and LLM parameters."""
        pass

    @abstractmethod
    def is_chat_mode(self) -> bool:
        """Returns True if the LLM is in chat mode, False otherwise."""
        pass

    def format_image(self, image: Image.Image) -> dict[str, Any]:
        """Returns a dictionary containing the specified image suitable for use in an LLM message."""
        raise NotImplementedError("This LLM does not support images.")

    async def generate_async(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        """Generates a response from the LLM for the given prompt and LLM parameters asynchronously."""
        raise NotImplementedError("This LLM does not support asynchronous generation.")

    def generate_batch(
        self, *, prompts: list[RenderedPrompt], llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> list[str]:
        """Generates a series of responses from the LLM for the given series of prompts. Order is preserved."""
        raise NotImplementedError("This LLM does not support batched generation")
