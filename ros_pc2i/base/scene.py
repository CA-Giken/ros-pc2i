# mypy: ignore-errors

from typing import Optional, Dict
from pyrr import Vector3
from .mesh import Mesh

class Scene:
    "シーン管理クラス"
    def __init__(self):
        self.meshes: Dict[str, Mesh] = {}
        self.light_position = Vector3([0.0, 0.0, 0.0])
        self.ambient_light = Vector3([0.1, 0.1, 0.1])

    def add_mesh(self, mesh: Mesh):
        self.meshes[mesh.name] = mesh
    
    def remove_mesh(self, name: str):
        self.meshes.pop(name)
    
    def get_mesh(self, name: str) -> Optional[Mesh]:
        return self.meshes.get(name)