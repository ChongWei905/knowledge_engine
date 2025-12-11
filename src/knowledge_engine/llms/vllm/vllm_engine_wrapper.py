import asyncio
import logging
import math
import time
from contextlib import nullcontext
from enum import Enum
from typing import Any, Dict

import asyncstdlib

from knowledge_engine.llms.prompts import RenderedMessage
from knowledge_engine.llms.vllm.vllm_engine_request import VLLMEngineRequest
from knowledge_engine.llms.vllm.vllm_output_data import VLLMOutputData

logger = logging.getLogger(__name__)


class VLLMTaskType(str, Enum):
    """The type of task to run on the vLLM engine."""

    """Generate text."""
    GENERATE = "generate"

    """Generate embeddings."""
    EMBED = "embed"


class VLLMEngineWrapper:
    def __init__(
        self,
        model: str,
        engine_kwargs: Dict[str, Any],
        idx_in_batch_column: str,
        task_type: VLLMTaskType = VLLMTaskType.GENERATE,
        max_output_len: int = 5120,
        max_pending_requests: int = -1,
        **kwargs,
    ):
        self.request_id: int = 0
        self.model = model
        self.engine_kwargs = engine_kwargs
        pp_size = self.engine_kwargs.get("pipeline_parallel_size", 1)
        self.max_pending_requests = max_pending_requests or math.ceil(
            self.engine_kwargs.get("max_num_seqs", 128) * pp_size * 1.1
        )
        self.idx_in_batch_column = idx_in_batch_column
        self.task_type = task_type

        try:
            import vllm
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed or failed to import. Please run "
                "`pip install ray[llm]` to install required dependencies."
            ) from e

        if self.task_type == VLLMTaskType.EMBED:
            raise ValueError("Embedding is not supported yet.")

        self.engine_args = vllm.AsyncEngineArgs(
            model=self.model,
            **self.engine_kwargs,
        )
        # create_engine_config will set default values including `max_num_seqs`.
        self.engine = vllm.AsyncLLMEngine.from_engine_args(self.engine_args)

        if self.max_pending_requests > 0:
            self.semaphore = asyncio.Semaphore(self.max_pending_requests)
        else:
            self.semaphore = asyncstdlib.nullcontext()

        self.sampling_params = vllm.SamplingParams(max_tokens=max_output_len)

        self._kwargs = kwargs

    async def generate_async(
        self,
        prompt: RenderedMessage,
        idx_in_batch: int = 0,
    ):
        request = await self._prepare_llm_request(prompt, idx_in_batch)
        t = time.perf_counter()

        async with self.semaphore:
            output = await self._generate_async(request)

        time_taken = time.perf_counter() - t

        output_data = VLLMOutputData.from_vllm_engine_output(output)
        return request, output_data.model_dump(), time_taken

    def shutdown(self):
        """Shutdown the vLLM v1 engine. This kills child processes forked
        by the vLLM engine. If not called, the child processes will be
        orphaned and will not be killed when the parent process exits,
        and they won't be able to be tracked by Ray anymore.
        """
        if hasattr(self.engine, "shutdown"):
            logger.info("Shutting down vLLM engine")
            self.engine.shutdown()

    def get_scheduler_config(self):
        return self._vllm_config.scheduler_config

    async def _prepare_llm_request(self, prompt: RenderedMessage, idx_in_batch: int) -> VLLMEngineRequest:
        """Prepare the inputs for LLM inference."""
        prompt_str = prompt.content
        tokenized_prompt = prompt.tokenized_content
        image = prompt.images

        if self.task_type == VLLMTaskType.GENERATE:
            params = self.sampling_params
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        request = VLLMEngineRequest(
            request_id=self.request_id,
            idx_in_batch=idx_in_batch,
            prompt=prompt_str,
            prompt_token_ids=tokenized_prompt,
            images=image,
            params=params,
        )
        self.request_id += 1
        return request

    async def _generate_async(self, request: VLLMEngineRequest) -> Any:
        """Process a single request.

        Args:
            request: The request.

        Returns:
            The output of the request.
        """

        # NOTE: vLLM v1 tighly couples tokenizer and detokenizer to the engine.
        # We should investigate whether decoupling them could lead to better
        # performance. Given that v1 tokenizer and detokenizer are already
        # in a separate process, the benefit of decoupling them in the Processor
        # may be limited.
        assert request.prompt
        import vllm

        multi_modal_data = {"image": request.images} if request.images else None
        llm_prompt = vllm.inputs.data.TextPrompt(
            prompt=request.prompt, multi_modal_data=multi_modal_data
        )

        # Send the request to the LLM engine.
        stream = self.engine.generate(
            request_id=str(request.request_id),
            prompt=llm_prompt,
            sampling_params=request.params,
        )

        # Consume the stream until the request is finished.
        async for request_output in stream:
            if request_output.finished:
                # Bypass the original full prompt.
                request_output.prompt = request.prompt
                return request_output

        raise RuntimeError(
            "[vLLM] The request is not finished. This should not happen. Please report this issue to the Ray team."
        )
