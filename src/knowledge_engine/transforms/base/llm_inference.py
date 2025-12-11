from abc import ABC, abstractmethod
from typing import Optional

from knowledge_engine.llms.llms import LLMFactory
from knowledge_engine.transforms import Node


class LLMInference(ABC):

    @abstractmethod
    def as_llm_map(
        self,
        child: Optional[Node],
        **kwargs,
    ):
        pass