import moderngl
from ros_pc2i.base.mesh import Mesh
from numpy.typing import NDArray
import numpy as np
from typing import Tuple

class Sphere(Mesh):
    """球体メッシュを生成するクラス"""
    def __init__(
        self,
        ctx: moderngl.Context,
        radius: float = 1.0,
        sectors: int = 32,
        stacks: int = 16,
        name: str = "sphere"
    ):
        """
        Parameters
        ----------
        ctx : moderngl.Context
            ModernGLコンテキスト
        radius : float, optional
            球の半径, by default 1.0
        sectors : int, optional
            経度方向の分割数, by default 32
        stacks : int, optional
            緯度方向の分割数, by default 16
        name : str, optional
            メッシュの名前, by default "sphere"
        """
        vertices, indices, normals = self._generate_sphere_mesh(radius, sectors, stacks)
        super().__init__(ctx, vertices, indices, normals, name)
    
    @staticmethod
    def _generate_sphere_mesh(
        radius: float,
        sectors: int,
        stacks: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32]]:
        """球体メッシュのジオメトリを生成する

        Parameters
        ----------
        radius : float
            球の半径
        sectors : int
            経度方向の分割数
        stacks : int
            緯度方向の分割数

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            頂点座標、インデックス、法線ベクトルの配列
        """
        vertices = []
        indices = []
        normals = []
        
        # 頂点座標と法線の生成
        for i in range(stacks + 1):
            V = i / stacks
            phi = V * np.pi
            
            for j in range(sectors + 1):
                U = j / sectors
                theta = U * 2 * np.pi
                
                # 球面座標から直交座標への変換
                x = radius * np.cos(theta) * np.sin(phi)
                y = radius * np.cos(phi)
                z = radius * np.sin(theta) * np.sin(phi)
                
                vertices.append([x, y, z])
                # 単位法線ベクトル
                normals.append([x/radius, y/radius, z/radius])
        
        # インデックスの生成
        for i in range(stacks):
            for j in range(sectors):
                first = i * (sectors + 1) + j
                second = first + sectors + 1
                
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])
        
        return (
            np.array(vertices, dtype=np.float32),
            np.array(indices, dtype=np.int32),
            np.array(normals, dtype=np.float32)
        )