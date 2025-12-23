
from typing import List

from pydantic import BaseModel, ConfigDict


# ==========================================
# 1. DATA STRUCTURES & HOOKS / СТРУКТУРИ ДАНИХ ТА ХУКИ
# ==========================================

class MatrixSteeringVector(BaseModel):
    """
    Data model for storing the trained steering matrix.

    Модель даних для зберігання навченої керуючої матриці.
    """
    layer_index: int
    intermediate_size: int
    active_indices: List[int]
    weight_matrix: List[List[float]]  # Shape: [N_active, Hidden_Size]
    bias_vector: List[float]  # Shape: [N_active]

    model_config = ConfigDict(arbitrary_types_allowed=True)