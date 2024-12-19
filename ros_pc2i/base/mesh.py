# mypy: ignore-errors

import struct
import moderngl
import numpy as np
from numpy.typing import NDArray

from .transform import Transform
from .material import Material

class Mesh:
    """
    メッシュデータを管理するクラス
    """
    def __init__(
      self,
      ctx: moderngl.Context,
      vertices: NDArray,
      normals: NDArray,
      name: str = "unnamed"
    ):
        self.ctx = ctx
        self.name = name
        self.vertices = vertices
        self.normals = normals
        # 頂点バッファの作成
        self.vao = None
        self.transform = Transform()
        self.material = Material()
        
    def update_vao(self, program: moderngl.Program):
        vertex_buffer = self.ctx.buffer(self.vertices.tobytes())
        normal_buffer = self.ctx.buffer(self.normals.tobytes())
        
        self.vao = self.ctx.vertex_array(
            program,
            [
                (vertex_buffer, '3f', 'in_position'),
                (normal_buffer, '3f', 'in_normal')
            ]
        )
    
    def render(self, program: moderngl.Program):
        if self.vao is None or self.vao.program != program:
            self.update_vao(program)
        self.vao.render()
        

def load_stl(
    ctx: moderngl.Context,
    filename
) -> Mesh:
    """
    STLファイルの読み込み
    
    Parameters:
    -----------
    filename : str
        STLファイルのパス
    """
    vertices = []
    normals = []

    with open(filename, 'rb') as f:
        f.seek(80) # スキップヘッダー
        num_triangles = struct.unpack('I', f.read(4))[0]

        for _ in range(num_triangles):
            data = struct.unpack('f' * 12 + 'H', f.read(50))
            normal = data[0:3]
            v1 = data[3:6]
            v2 = data[6:9]
            v3 = data[9:12]

            vertices.extend([v1, v2, v3])
            normals.extend([normal] * 3)

    mesh = Mesh(
        ctx=ctx,
        vertices=np.array(vertices, dtype='f4'),
        normals=np.array(normals, dtype='f4')
    )

    return mesh