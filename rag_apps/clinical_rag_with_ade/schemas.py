from pydantic import BaseModel, Field
from typing import List, Optional

class BoundingBox(BaseModel):
    left: float
    top: float
    right: float
    bottom: float

class ClinicalChunk(BaseModel):
    chunk_id: str
    text: str
    page: int
    chunk_type: str
    bbox: Optional[BoundingBox] = None

    class Config:
        from_attributes = True