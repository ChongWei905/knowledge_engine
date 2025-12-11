
import asyncio
import logging
import uuid
from typing import Optional, Any, Dict

from transformers import AutoProcessor

from knowledge_engine.llms.llms import LLM
from knowledge_engine.llms.config import LLMModel, LLMMode
from knowledge_engine.llms.prompts import RenderedPrompt, RenderedMessage
from knowledge_engine.llms.vllm.vllm_engine_wrapper import VLLMEngineWrapper, VLLMTaskType
from knowledge_engine.llms.vllm.vllm_utils import get_prompt

logger = logging.getLogger(__name__)




class VLLM(LLM):
    """VLLM implementation for local LLM inference using VLLMEngineWrapper."""

    def __init__(
        self,
        model_name: str,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        max_output_len: int = 5120,
        max_pending_requests: int = -1,
        default_mode: LLMMode = LLMMode.ASYNC,
        vllm_task_type: VLLMTaskType = VLLMTaskType.GENERATE,
        default_llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize VLLM with VLLMEngineWrapper.

        Args:
            model_name: The name or path of the model to load
            engine_kwargs: Additional keyword arguments to pass to vLLM engine
            max_output_len: Maximum number of tokens to generate
            max_pending_requests: Maximum number of pending requests (-1 for unlimited)
            default_mode: Default execution mode (GENERATE or EMBED)
            default_llm_kwargs: Default LLM parameters
            **kwargs: Additional arguments
        """
        super().__init__(
            model_name=model_name,
            default_mode=default_mode,
            default_llm_kwargs=default_llm_kwargs,
        )

        # Initialize engine kwargs with defaults
        self.engine_kwargs = engine_kwargs or {}
        if "tensor_parallel_size" not in self.engine_kwargs:
            self.engine_kwargs["tensor_parallel_size"] = 1
        if "max_num_seqs" not in self.engine_kwargs:
            self.engine_kwargs["max_num_seqs"] = 128

        # Determine task type from mode
        self.vllm_task_type = vllm_task_type

        # Initialize the wrapper
        self.wrapper = VLLMEngineWrapper(
            model=model_name,
            engine_kwargs=self.engine_kwargs,
            idx_in_batch_column="idx_in_batch",
            task_type=self.vllm_task_type,
            max_output_len=max_output_len,
            max_pending_requests=max_pending_requests,
            **kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        logger.info(f"Initialized VLLM with model: {model_name}")

    def is_chat_mode(self) -> bool:
        """VLLM uses completion mode"""
        return False

    def generate(
        self,
        *,
        prompt: RenderedPrompt,
        llm_kwargs: Optional[dict] = None,
        model: Optional[LLMModel] = None,
    ) -> str:
        """
        Generates a response from the LLM for the given prompt synchronously.

        Args:
            prompt: The rendered prompt to generate from
            llm_kwargs: Optional LLM parameters
            model: Optional model override (not used in this implementation)

        Returns:
            Generated text string
        """
        # Run async generation in a sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If already in an async context, create a new loop
            import nest_asyncio
            nest_asyncio.apply()
            result = loop.run_until_complete(
                self.generate_async(prompt=prompt, llm_kwargs=llm_kwargs, model=model)
            )
        else:
            result = loop.run_until_complete(
                self.generate_async(prompt=prompt, llm_kwargs=llm_kwargs, model=model)
            )

        return result

    async def generate_async(
        self,
        *,
        prompt: RenderedPrompt,
        llm_kwargs: Optional[dict] = None,
        model: Optional[LLMModel] = None,
    ) -> str:
        """
        Generates a response from the LLM for the given prompt asynchronously.

        Args:
            prompt: The rendered prompt to generate from
            llm_kwargs: Optional LLM parameters
            model: Optional model override (not used in this implementation)

        Returns:
            Generated text string
        """
        # Merge LLM kwargs
        merged_kwargs = self._merge_llm_kwargs(llm_kwargs)

        # Update sampling params if provided
        if merged_kwargs:
            self._update_sampling_params(merged_kwargs)

        # Convert RenderedPrompt to RenderedMessage
        rendered_message = self._convert_prompt_to_message(prompt)

        # Generate using the wrapper
        request, output_data, time_taken = await self.wrapper.generate_async(
            prompt=rendered_message, idx_in_batch=0
        )

        logger.debug(f"Generation took {time_taken:.2f}s")

        # Extract generated text from output
        generated_text = output_data.get("generated_text", "")
        return generated_text

    def generate_batch(
        self,
        *,
        prompts: list[RenderedPrompt],
        llm_kwargs: Optional[dict] = None,
        model: Optional[LLMModel] = None,
    ) -> list[str]:
        """
        Generates a series of responses from the LLM for the given series of prompts.

        Args:
            prompts: List of rendered prompts to generate from
            llm_kwargs: Optional LLM parameters
            model: Optional model override (not used in this implementation)

        Returns:
            List of generated text strings (order preserved)
        """

        messages = [self._convert_prompt_to_message(prompt) for prompt in prompts]

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an async context, use nest_asyncio
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Define the async batch function
        async def _batch_generate_async():
            # Create tasks inside the async function
            tasks = [
                self.wrapper.generate_async(message, idx)
                for idx, message in enumerate(messages)
            ]
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            return results

        # Run the async function
        results = loop.run_until_complete(_batch_generate_async())

        # Process results and sort by batch index
        outputs = []
        for request, output_data, time_taken in results:
            outputs.append({
                "idx_in_batch": request.idx_in_batch,
                "generated_text": output_data.get("generated_text", "")
            })

        # Sort by index to preserve order
        outputs.sort(key=lambda x: x["idx_in_batch"])

        return [output["generated_text"] for output in outputs]

    def _convert_prompt_to_message(self, prompt: RenderedPrompt) -> RenderedMessage:
        """
        Convert RenderedPrompt to RenderedMessage for the wrapper.

        Args:
            prompt: The rendered prompt to convert

        Returns:
            RenderedMessage object
        """
        # Create a basic RenderedMessage
        # Assuming RenderedPrompt has a 'content' or similar attribute
        messages = prompt.messages
        if len(messages) > 1:
            logger.warning(f"VLLM only supports a single message, using the first message")
        message = messages[0]
        message.content = get_prompt(self.processor, message.content, bool(message.images))

        return message

    def _update_sampling_params(self, llm_kwargs: dict):
        """
        Update the wrapper's sampling parameters.

        Args:
            llm_kwargs: LLM parameters to update
        """
        # Update sampling params if specific keys are provided
        import vllm

        sampling_kwargs = {}
        if "temperature" in llm_kwargs:
            sampling_kwargs["temperature"] = llm_kwargs["temperature"]
        if "top_p" in llm_kwargs:
            sampling_kwargs["top_p"] = llm_kwargs["top_p"]
        if "top_k" in llm_kwargs:
            sampling_kwargs["top_k"] = llm_kwargs["top_k"]
        if "max_tokens" in llm_kwargs:
            sampling_kwargs["max_tokens"] = llm_kwargs["max_tokens"]

        if sampling_kwargs:
            self.wrapper.sampling_params = vllm.SamplingParams(**sampling_kwargs)

    def shutdown(self):
        """Shutdown the vLLM engine."""
        self.wrapper.shutdown()
        logger.info("VLLM engine shut down successfully")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"Error during VLLM cleanup: {e}")

