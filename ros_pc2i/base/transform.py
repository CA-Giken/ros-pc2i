#mypy: ignore-errors

import numpy as np
from pyrr import Matrix44, Vector3, matrix44
from dataclasses import dataclass

@dataclass
class Transform:
    """3D変換を管理するクラス"""
    def __init__(self):
        self.position = Vector3([0.0, 0.0, 0.0])
        self.rotation = Vector3([0.0, 0.0, 0.0]) # Euler角
        self.scale = Vector3([1.0, 1.0, 1.0])

    def get_model_matrix(self) -> np.ndarray:
        """モデル行列の計算"""
        translation = Matrix44.from_translation(self.position)
        rotation = matrix44.create_from_eulers(self.rotation)
        scale = Matrix44.from_scale(self.scale)
        return translation * rotation * scale