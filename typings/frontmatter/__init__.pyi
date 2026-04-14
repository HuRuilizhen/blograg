from typing import Any

class Post:
    metadata: dict[Any, Any]
    content: str

def load(fd: str) -> Post: ...
