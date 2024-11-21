#!/usr/bin/env python
# coding: utf-8
# mypy: ignore-errors

import moderngl
import numpy as np
from pyrr import Vector3, Matrix44, Quaternion, vector
from stl import mesh
import glm
from PIL import Image
import pygame
import sys

class Scene:
    gl_version = (3, 3)

    def __init__(self, width=800, height=600):
        # ModernGLコンテキストの初期化
        self.ctx = moderngl.create_standalone_context()
        self.width = width
        self.height = height
        
        # シェーダーの設定
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330
                
                uniform mat4 model;
                uniform mat4 view;
                uniform mat4 projection;
                
                in vec3 in_position;
                in vec3 in_normal;
                
                out vec3 v_normal;
                out vec3 v_position;
                
                void main() {
                    v_normal = mat3(transpose(inverse(model))) * in_normal;
                    vec4 world_position = model * vec4(in_position, 1.0);
                    v_position = world_position.xyz;
                    gl_Position = projection * view * world_position;
                }
            ''',
            fragment_shader='''
                #version 330
                
                struct Material {
                    vec3 ambient;
                    vec3 diffuse;
                    vec3 specular;
                    float shininess;
                    float roughness;
                    float metallic;
                };
                
                struct Light {
                    vec3 position;
                    vec3 ambient;
                    vec3 diffuse;
                    vec3 specular;
                };
                
                uniform Material material;
                uniform Light light;
                uniform vec3 viewPos;

                in vec3 v_normal;
                in vec3 v_position;
                in vec3 v_view;

                out vec4 f_color;

                void main() {
                    // アンビエント
                    vec3 ambient = light.ambient * material.ambient;

                    // ディフューズ
                    vec3 norm = normalize(v_normal);
                    vec3 lightDir = normalize(light.position - v_position);
                    float diff = max(dot(norm, lightDir), 0.0);
                    vec3 diffuse = light.diffuse * (diff * material.diffuse);

                    // スペキュラー
                    vec3 viewDir = normalize(viewPos - v_position);
                    vec3 reflectDir = reflect(-lightDir, norm);
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
                    vec3 specular = light.specular * (spec * material.specular);

                    // メタリック
                    vec3 specColor = mix(vec3(0.04), material.specular, material.metallic);
                    vec3 result = ambient + diffuse + specular;
                    f_color = vec4(result, 1.0);
                }
            '''
        )
        
        # カメラ
        self.camera = CameraController(width / height)

        # フレームバッファの設定
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def load_stl(self, stl_path):
        # STLファイルを読み込み、頂点とインデックスデータを生成
        mesh_data = mesh.Mesh.from_file(stl_path)
        
        vertices = []
        normals = []
        indices = []
        
        for i, triangle in enumerate(mesh_data.vectors):
            for vertex in triangle:
                vertices.append(vertex)
                normals.append(mesh_data.normals[i])
            indices.extend([i * 3, i * 3 + 1, i * 3 + 2])
        
        vertices = np.array(vertices, dtype='f4')
        normals = np.array(normals, dtype='f4')
        indices = np.array(indices, dtype='i4')
        
        # VBOとVAOの設定
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.nbo = self.ctx.buffer(normals.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '3f', 'in_position'),
                (self.nbo, '3f', 'in_normal'),
            ],
            self.ibo
        )
        
        return len(indices)
    
    def render(self, camera_pos, target_pos, material_props=None, light_props=None):
        # デフォルトのマテリアルとライトプロパティ
        default_material = {
            'ambient': (0.1, 0.1, 0.1),
            'diffuse': (0.7, 0.7, 0.7),
            'specular': (1.0, 1.0, 1.0),
            'shininess': 32.0
        }
        
        default_light = {
            'position': (5.0, 5.0, 5.0),
            'ambient': (0.2, 0.2, 0.2),
            'diffuse': (0.5, 0.5, 0.5),
            'specular': (1.0, 1.0, 1.0)
        }
        
        material = {**default_material, **(material_props or {})}
        light = {**default_light, **(light_props or {})}
        
        # ビュー行列とプロジェクション行列の設定
        view = glm.lookAt(
            glm.vec3(*camera_pos),
            glm.vec3(*target_pos),
            glm.vec3(0, 1, 0)
        )
        
        projection = glm.perspective(
            glm.radians(45.0),
            self.width / self.height,
            0.1,
            100.0
        )
        
        # ユニフォーム変数の設定
        # self.program["camera"].write(self.camera)
        self.program['model'].write(glm.mat4(1.0).to_bytes())
        self.program['view'].write(view.to_bytes())
        self.program['projection'].write(projection.to_bytes())
        self.program['viewPos'].write(glm.vec3(*camera_pos).to_bytes())
        
        # マテリアルプロパティの設定
        for key, value in material.items():
            if isinstance(value, (float, int)):
                self.program[f'material.{key}'] = value
            else:
                self.program[f'material.{key}'].write(glm.vec3(*value).to_bytes())
        
        # ライトプロパティの設定
        for key, value in light.items():
            self.program[f'light.{key}'].write(glm.vec3(*value).to_bytes())
        
        # レンダリング
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render()
        
        # ピクセルデータを返す
        return self.fbo.read(components=4)
    
    def cleanup(self):
        self.vao.release()
        self.vbo.release()
        self.nbo.release()
        self.ibo.release()
        self.program.release()
        self.fbo.release()
        self.ctx.release()



class CameraController:
    def __init__(self, ratio):
        self._zoom_step = 0.1
        self._move_vertically = 0.1
        self._move_horizontally = 0.1
        self._rotate_horizontally = 0.1
        self._rotate_vertically = 0.1

        self._field_of_view_degrees = 60.0
        self._z_near = 0.1
        self._z_far = 100
        self._ratio = ratio
        self.build_projection()

        self._camera_position = Vector3([0.0, 0.0, -40.0])
        self._camera_front = Vector3([0.0, 0.0, 1.0])
        self._camera_up = Vector3([0.0, 1.0, 0.0])
        self._cameras_target = (self._camera_position + self._camera_front)
        self.build_look_at()

    def zoom_in(self):
        self._field_of_view_degrees = self._field_of_view_degrees - self._zoom_step
        self.build_projection()

    def zoom_out(self):
        self._field_of_view_degrees = self._field_of_view_degrees + self._zoom_step
        self.build_projection()

    def move_forward(self):
        self._camera_position = self._camera_position + self._camera_front * self._move_horizontally
        self.build_look_at()

    def move_backwards(self):
        self._camera_position = self._camera_position - self._camera_front * self._move_horizontally
        self.build_look_at()

    def strafe_left(self):
        self._camera_position = self._camera_position - vector.normalize(self._camera_front ^ self._camera_up) * self._move_horizontally
        self.build_look_at()

    def strafe_right(self):
        self._camera_position = self._camera_position + vector.normalize(self._camera_front ^ self._camera_up) * self._move_horizontally
        self.build_look_at()

    def strafe_up(self):
        self._camera_position = self._camera_position + self._camera_up * self._move_vertically
        self.build_look_at()

    def strafe_down(self):
        self._camera_position = self._camera_position - self._camera_up * self._move_vertically
        self.build_look_at()

    def rotate_left(self):
        rotation = Quaternion.from_y_rotation(2 * float(self._rotate_horizontally) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def rotate_right(self):
        rotation = Quaternion.from_y_rotation(-2 * float(self._rotate_horizontally) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def build_look_at(self):
        self._cameras_target = (self._camera_position + self._camera_front)
        self.mat_lookat = Matrix44.look_at(
            self._camera_position,
            self._cameras_target,
            self._camera_up)

    def build_projection(self):
        self.mat_projection = Matrix44.perspective_projection(
            self._field_of_view_degrees,
            self._ratio,
            self._z_near,
            self._z_far)

def render_stl_to_image(stl_path, camera_pos, output_path, width=800, height=600, 
                       material_props=None, light_props=None):
    """
    STLファイルをレンダリングしてPNG画像として保存する
    
    Parameters:
    -----------
    stl_path : str
        入力STLファイルのパス
    camera_pos : tuple
        カメラの位置 (x, y, z)
    output_path : str
        出力PNG画像のパス
    width : int
        出力画像の幅
    height : int
        出力画像の高さ
    material_props : dict
        マテリアル設定（オプション）
    light_props : dict
        ライト設定（オプション）
    """
    # レンダラーの初期化
    renderer = Scene(width, height)
    renderer.load_stl(stl_path)
    
    # デフォルトのマテリアル設定
    default_material = {
        'ambient': (0.2, 0.2, 0.2),
        'diffuse': (0.7, 0.7, 0.7),
        'specular': (1.0, 1.0, 1.0),
        'shininess': 32.0
    }
    
    # デフォルトのライト設定
    default_light = {
        'position': (10.0, 10.0, 10.0),
        'ambient': (0.2, 0.2, 0.2),
        'diffuse': (0.8, 0.8, 0.8),
        'specular': (1.0, 1.0, 1.0)
    }
    
    # ユーザー設定があれば上書き
    material = {**default_material, **(material_props or {})}
    light = {**default_light, **(light_props or {})}
    
    try:
        # レンダリング実行
        pixels = renderer.render(
            camera_pos=camera_pos,
            target_pos=(0, 0, 0),  # 原点を注視点とする
            material_props=material,
            light_props=light
        )
        
        # ピクセルデータを画像に変換
        image = Image.frombytes('RGBA', (width, height), pixels)
        
        # 画像の上下を反転（OpenGLの座標系に合わせる）
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # PNG形式で保存
        image.save(output_path, 'PNG')
        

        # スタッツ表示
        pixel_data = np.array(image)
        pixel_stats = {
            'min': pixel_data.min(),
            'max': pixel_data.max(),
            'mean': pixel_data.mean(),
            'non_zero': np.count_nonzero(pixel_data)
        }
        print("\nレンダリング統計:")
        print(f"最小値: {pixel_stats['min']}")
        print(f"最大値: {pixel_stats['max']}")
        print(f"平均値: {pixel_stats['mean']:.2f}")
        print(f"非ゼロピクセル数: {pixel_stats['non_zero']}")
        print(f"デバッグ画像保存先: {output_path}")
        
        print("✓ レンダリング出力に成功")
    finally:
        # リソースの解放
        renderer.cleanup()

def open_viewer(filepath):
    pygame.init()
    pygame.display.set_mode((800, 600), pygame.DOUBLEBUF | pygame.OPENGL)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            break
    
    scene = Scene(800, 600)
    scene.load_stl(filepath)
    scene.render((-10, -10, -10), (0, 0, 0))
    pygame.display.flip()

# 使用例
if __name__ == "__main__":
    # マテリアル設定の例
    metal_material = {
        'ambient': (0.2, 0.2, 0.2),
        'diffuse': (0.8, 0.8, 0.8),
        'specular': (1.0, 1.0, 1.0),
        'shininess': 64.0
    }

    # ビュワー起動
    open_viewer('../data/hole.stl')