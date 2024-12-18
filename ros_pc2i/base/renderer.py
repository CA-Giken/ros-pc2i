# mypy: ignore-errors

import os
from pathlib import Path
import numpy as np
from PIL import Image
import glfw
import moderngl as mgl
from .scene import Scene
from .camera import Camera

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

class Renderer:
    "レンダラークラス"
    def __init__(self, width: int = 800, height: int = 600, headless=False):
        # ウィンドウの初期化
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        if headless:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(width, height, "Renderer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        self.ctx = mgl.create_context(standalone=False)
        # ModernGLの初期化
        self.width = width
        self.height = height

        shader_dir = Path(__file__).parent.parent / "shaders"
        self.prog = self.load_shaders(
            shader_dir / "vertex.glsl",
            shader_dir / "fragment.glsl"
        )

        # フレームバッファの設定
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)],
            depth_attachment=self.ctx.depth_texture((width, height))
        )

        # OpenGLの設定
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.CULL_FACE)

    def __del__(self):
        glfw.terminate()

    def load_shaders(self, vertex_path: Path, fragment_path: Path):
        with open(vertex_path, "r") as f:
            vertex_shader = f.read()
        with open(fragment_path, "r") as f:
            fragment_shader = f.read()
        return self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

    def render(self, scene: Scene, camera: Camera):
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0)

        # ビュー行列の作成
        # Z軸方向を見るようにカメラを設定
        view = np.eye(4)
        view[:3, :3] = camera.R.T
        view[:3, 3] = -camera.R.T @ camera.t.flatten()
        # MVP (Model-View-Projection) 行列の計算
        projection = camera.compute_projection_matrix(self.width, self.height)
        mvp = projection @ view

        self.prog['mvp'].write(mvp.astype('f4').tobytes())
        self.prog['light_position'].write(np.array(scene.light_position, dtype='f4').tobytes())
        self.prog['ambient_light'].write(np.array(scene.ambient_light, dtype='f4').tobytes())

        # メッシュの描画
        for mesh_name in scene.meshes.keys():
            mesh = scene.get_mesh(mesh_name)
            if mesh is None:
                continue

            model_matrix = mesh.transform.get_model_matrix()
            self.prog["model_matrix"].write(model_matrix.astype('f4').tobytes())
            self.prog['camera_position'].write(camera.t[:, 0].astype('f4').tobytes())

            # マテリアルの設定
            self.prog['material.color'].write(np.array(mesh.material.color, dtype='f4'))
            self.prog['material.ambient'].write(np.array(mesh.material.ambient, dtype='f4'))
            self.prog['material.diffuse'].write(np.array(mesh.material.diffuse, dtype='f4'))
            self.prog['material.specular'].write(np.array(mesh.material.specular, dtype='f4'))
            self.prog['material.shininess'].write(np.array(mesh.material.shininess, dtype='f4'))
            self.prog['material.metallic'].write(np.array(mesh.material.metallic, dtype='f4'))
            self.prog['material.roughness'].write(np.array(mesh.material.roughness, dtype='f4'))
            mesh.render(self.prog)

        image = Image.frombytes("RGBA", self.fbo.size, self.fbo.read(components=4))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        # 深度情報はアルファチャンネルに格納されている
        depth_map = np.array(image)[:, :, 3]
        return image, depth_map