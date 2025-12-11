from enum import Enum


class LLMMode(Enum):
    SYNC = 1
    ASYNC = 2
    BATCH = 3

class LLMModel:
    name: str
    is_chat: bool
