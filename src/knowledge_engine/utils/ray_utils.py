from pathlib import Path

class RayPathParser:
    def __call__(self, path: str) -> Path:
        return Path(path)