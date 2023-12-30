from enum import StrEnum
from typing import Any, Protocol, TypeAlias, TypedDict, TypeGuard

import numpy as np
from pydantic import BaseModel

ndarray_f32: TypeAlias = np.ndarray[int, np.dtype[np.float32]]
ndarray_i64: TypeAlias = np.ndarray[int, np.dtype[np.int64]]
ndarray_i32: TypeAlias = np.ndarray[int, np.dtype[np.int32]]


class TextResponse(BaseModel):
    __root__: str


class MessageResponse(BaseModel):
    message: str


class BoundingBox(TypedDict):
    x1: int
    y1: int
    x2: int
    y2: int


class ModelType(StrEnum):
    CLIP = "clip"
    FACIAL_RECOGNITION = "facial-recognition"


class HasProfiling(Protocol):
    profiling: dict[str, float]


class Face(TypedDict):
    boundingBox: BoundingBox
    embedding: ndarray_f32
    imageWidth: int
    imageHeight: int
    score: float


class ClusterRequest(BaseModel):
    embeddings: list[list[float]]
    min_cluster_size: int = 5
    min_samples: int = None
    cluster_selection_epsilon: float = 0.0
    max_cluster_size: int = 0
    metric: str = "euclidean"
    alpha: float = 1.0
    algorithm: str = "best"
    leaf_size: int = 40
    approx_min_span_tree: bool = True
    cluster_selection_method: str = "eom"
    
    class Config:
        arbitrary_types_allowed = True


def has_profiling(obj: Any) -> TypeGuard[HasProfiling]:
    return hasattr(obj, "profiling") and isinstance(obj.profiling, dict)
