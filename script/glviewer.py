#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors

import moderngl
import numpy as np
import struct
from PIL import Image
import glfw
from pyrr import Matrix44, Vector3

class Scene:
    def __init__(self, config):
        self.config = {
            "resolution": (800, 600),

        }
        # GLFW init
        if not glfw.init():
            raise
        self.window = glfw.create_window(self.config["resolution"][0], self.config["resolution"][1], "Scene", None, None)
        if not self.window:
            glfw.terminate()
            raise
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()
        
        self.program = self.ctx.program(
          vertex_shader='''
              #version 330
              
              uniform mat4 mvp;
              uniform mat4 model;
              uniform vec3 cameraPos;
              
              in vec3 in_position;
              in vec3 in_normal;
              
              out vec3 v_normal;
              out vec3 v_position;
              out vec3 v_view;
              
              void main() {
                  vec4 worldPos = model * vec4(in_position, 1.0);
                  gl_Position = mvp * vec4(in_position, 1.0);
                  
                  v_normal = mat3(model) * in_normal;
                  v_position = worldPos.xyz;
                  v_view = cameraPos - worldPos.xyz;
              }
          ''',
          fragment_shader='''
              #version 330
              
              in vec3 v_normal;
              in vec3 v_position;
              in vec3 v_view;
              
              out vec4 fragColor;
              
              // メタルマテリアルのパラメータ
              const vec3 metalColor = vec3(0.95, 0.93, 0.88);    // 金属の基本色
              const float roughness = 0.3;                        // 粗さ（小さいほど鏡面反射が強い）
              const float metallic = 0.9;                         // 金属度
              const float ambient = 0.2;                          // 環境光
              
              void main() {
                  vec3 N = normalize(v_normal);
                  vec3 V = normalize(v_view);
                  
                  // 主光源の設定
                  vec3 lightPos = vec3(50.0, 50.0, 50.0);
                  vec3 L = normalize(lightPos - v_position);
                  
                  // 反射ベクトル
                  vec3 R = reflect(-L, N);
                  
                  // フレネル効果（視線角度による反射率の変化）
                  float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0);
                  
                  // 鏡面反射
                  float specular = pow(max(dot(R, V), 0.0), 1.0/roughness);
                  
                  // 拡散反射
                  float diffuse = max(dot(N, L), 0.0);
                  
                  // メタリックワークフロー
                  vec3 specColor = mix(vec3(0.04), metalColor, metallic);
                  vec3 diffuseColor = mix(metalColor, vec3(0.0), metallic);
                  
                  // 最終的な色の計算
                  vec3 finalColor = 
                      diffuseColor * diffuse * (1.0 - fresnel) +
                      specColor * specular * (fresnel + metallic) +
                      ambient * metalColor;
                  
                  // HDRトーンマッピング
                  finalColor = finalColor / (finalColor + vec3(1.0));
                  
                  fragColor = vec4(finalColor, 1.0);
              }
          '''
        )
        
        # フレームバッファの設定
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture(self.config["resolution"], 4)],
            depth_attachment=self.ctx.depth_texture(self.config["resolution"])
        )
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def render(self, camera_pos, target_pos):
        view = Matrix44.look_at(
            camera_pos,
            target_pos,
            (0, 1, 0),
        )
        projection = Matrix44.perspective_projection(45.0, self.config["resolution"][0]/self.config["resolution"][1], 0.1, 1000.0)

        rotation = 0.1
        model = Matrix44.from_y_rotation(rotation)
        mvp = projection * view * model
        
        self.program['mvp'].write(mvp.astype('f4').tobytes())
        self.program['model'].write(model.astype('f4').tobytes())
        self.program['cameraPos'].write(camera_pos.astype('f4').tobytes())
        
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render()
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()

        # return pixels
        return self.fbo.read(components=4)
        

    def load_stl(self, filename):
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
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.nbo = self.ctx.buffer(normals.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '3f', 'in_position'),
                (self.nbo, '3f', 'in_normal'),
            ]
        )

    def capture_snapshot(self, filepath):
        width, height = self.config["resolution"]
        pixels = self.fbo.read(components=4)
        image = Image.frombytes('RGBA', (width, height), pixels)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(filepath)
        print(f"Snapshot saved to {filepath}")

    def cleanup(self):
        self.vao.release()
        self.vbo.release()
        self.nbo.release()
        self.fbo.release()
        self.program.release()
        self.ctx.release()
        glfw.terminate()

if __name__ == '__main__':
    config = {"resolution": (800, 600)}
    scene = Scene(config=config)
    scene.load_stl('../data/hole.stl')
    camera_pos = Vector3([200.0, 200.0, 200.0])
    target_pos = Vector3([0.0, 0.0, 0.0])
    while True:
        scene.render(camera_pos=camera_pos, target_pos=target_pos)
    scene.capture_snapshot('../data/snapshot.png')
    scene.cleanup()