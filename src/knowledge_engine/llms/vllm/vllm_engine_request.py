from typing import List, Any, Optional

from pydantic import BaseModel


class VLLMEngineRequest(BaseModel):
    """A request to the vLLM engine."""

    # The request ID for the LLM engine (unique per replica).
    request_id: int
    # The index of the request in the batch.
    idx_in_batch: int
    # The full prompt string (with chat template applied if any).
    prompt: str
    # The images inputs for the multimodal model. Use Any to avoid importing PIL.
    images: Optional[List[Any]]
    # The tokenized prompt IDs. If None, then the LLM engine will
    # tokenize the string prompt. This is not recommended for performance reasons.
    prompt_token_ids: Optional[List[int]]
    # The sampling or pooling parameters. Use Any to avoid importing vLLM.
    params: Any

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True