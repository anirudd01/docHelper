from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Org:
    id: Optional[int]
    name: str
    is_active: bool = True


@dataclass
class PDF:
    id: Optional[int]
    org_id: int
    filename: str
    chunk_size: int
    upload_time: Optional[str] = None
    is_active: bool = True


@dataclass
class Chunk:
    id: Optional[int]
    pdf_id: int
    chunk_index: int
    text: str


@dataclass
class Embedding:
    id: Optional[int]
    chunk_id: int
    embedding: List[float]
