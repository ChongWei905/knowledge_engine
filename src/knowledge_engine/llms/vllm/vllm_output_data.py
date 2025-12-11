import dataclasses
from typing import Optional, List, Any, Dict

import numpy as np
from pydantic import BaseModel, Field


class VLLMOutputData(BaseModel):
    """The output of the vLLM engine."""

    prompt: str
    prompt_token_ids: Optional[List[int]]
    num_input_tokens: int

    # Generate fields.
    generated_tokens: List[int] = Field(default_factory=list)
    generated_text: str = Field(default="")
    num_generated_tokens: int = Field(default=0)

    # Embed fields. The type should be torch.Tensor, but we use Any to avoid
    # importing torch because of an error in sphinx-build with an unknown reason.
    embeddings: Optional[Any] = None

    # Metrics fields.
    metrics: Optional[Dict[str, Any]] = None

    @classmethod
    def from_vllm_engine_output(cls, output: Any) -> "VLLMOutputData":
        """Create a vLLMOutputData from a vLLM engine output."""

        prompt_token_ids = output.prompt_token_ids
        if isinstance(prompt_token_ids, np.ndarray):
            prompt_token_ids = prompt_token_ids.tolist()

        data = cls(
            prompt=output.prompt,
            prompt_token_ids=prompt_token_ids,
            num_input_tokens=len(prompt_token_ids),
        )

        import vllm

        if isinstance(output, vllm.outputs.RequestOutput):
            metrics = {}
            if output.metrics is not None:
                metrics = dict(dataclasses.asdict(output.metrics))
                data.metrics = metrics
            data.generated_tokens = output.outputs[0].token_ids
            data.generated_text = output.outputs[0].text
            data.num_generated_tokens = len(output.outputs[0].token_ids)
        else:
            raise ValueError(f"Unknown output type: {type(output)}")

        return data

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True