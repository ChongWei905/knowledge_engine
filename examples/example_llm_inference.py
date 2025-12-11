import logging
from typing import Optional

from knowledge_engine.llms.llms import LLMFactory
from knowledge_engine.llms.vllm.prompt_processors import DocumentTextProcessor
from knowledge_engine.transforms import Node
from knowledge_engine.transforms.base.llm_inference import LLMInference
from knowledge_engine.transforms.base.llm_map import LLMMap

logger = logging.getLogger(__name__)


class ExampleLLMInference(LLMInference):
    def __init__(
        self,
        llm_factory: LLMFactory,
        tokenize: bool = False,
        add_generation_prompt: bool = True
    ):
        self._llm_factory = llm_factory
        self._tokenize = tokenize
        self._add_generation_prompt = add_generation_prompt
        pass

    def as_llm_map(
        self,
        child: Optional[Node] = None,
        **kwargs,
    ):
        prompt = DocumentTextProcessor(
            tokenize=self._tokenize, add_generation_prompt=self._add_generation_prompt)

        node = LLMMap(
            child=child,
            prompt=prompt,
            output_field="generated_text",
            llm_factory=self._llm_factory,
            validate=lambda d: d.properties.get("generated_text", None) is not None,
            iteration_var="generated_text_i",
            max_tries=100,
            **kwargs,
        )
        return node





















