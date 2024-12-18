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
      indices: NDArray,
      normals: NDArray,
      name: str = "unnamed"
    ):
        self.ctx = ctx
        self.name = name
        
        # 頂点バッファの作成
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.nbo = self.ctx.buffer(normals.astype('f4').tobytes())
        self.index_count = len(indices)
        
        self.vao = None
        self.transform = Transform()
        self.material = Material()
        
    def update_vao(self, program: moderngl.Program):
        self.vao = self.ctx.vertex_array(
            program,
            [
                (self.vbo, '3f', 'in_position'),
                (self.nbo, '3f', 'in_normal'),
            ],
            self.ibo,
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
        f.seek(80)
        num_triangles = struct.unpack('I', f.read(4))[0]
        
        for _ in range(num_triangles):
            nx, ny, nz = struct.unpack('fff', f.read(12))
            for _ in range(3):
                x, y, z = struct.unpack('fff', f.read(12))
                vertices.extend([x, y, z])
                normals.extend([nx, ny, nz])
            f.seek(2, 1)
            
    vertices = np.array(vertices, dtype='f4')
    normals = np.array(normals, dtype='f4')

    mesh = Mesh(ctx=ctx, vertices=vertices, indices=np.arange(len(vertices)//3), normals=normals)
    
    return mesh