#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors

from pathlib import Path
import struct
import moderngl
import numpy as np
from PIL import Image
import glfw
from pyrr import Matrix44, matrix44
import yaml

def cal_fov(focal_length, sensor_size):
    """
    焦点距離とセンサーサイズから画角を計算
    
    Parameters:
    -----------
    focal_length : float
        焦点距離
    sensor_size : float
        センサーサイズ
        
    Returns:
    --------
    float : 画角
    """
    return 2 * np.arctan(sensor_size / (2 * focal_length))

class CameraParameters:
    def __init__(self, fx, fy, cx, cy, R=None, t=None):
        """
        カメラパラメータを管理するクラス
        
        Parameters:
        -----------
        fx, fy : float
            x方向、y方向の焦点距離
        cx, cy : float
            主点座標
        R : ndarray, shape (3, 3)
            回転行列（デフォルトは単位行列）
        t : ndarray, shape (3, 1)
            並進ベクトル（デフォルトは零ベクトル）
        """
        # 内部パラメータ行列（K行列）の構築
        self.K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ])
        
        # 外部パラメータの設定
        self.R = np.eye(3) if R is None else R
        self.t = np.zeros((3, 1)) if t is None else t
        
    def compute_projection_matrix(self, width, height, near=0.1, far=1000):
        """
        射影行列の計算
        
        Returns:
        --------

        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        # fovy = cal_fov(fy, height)
        # aspect = width / height
        
        # projection = Matrix44.perspective_projection(
        #     fovy,   # y方向の視野角
        #     aspect, # アスペクト比
        #     near,   # ニアクリップ面
        #     far,    # ファークリップ面
        # )

        # # 主点オフセットの補正行列を作成
        # # OpenGLの正規化デバイス座標系に合わせて補正
        # offset_x = (2 * cx / width - 1)
        # offset_y = (2 * cy / height - 1)
        # correction = Matrix44([
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 0],
        #     [offset_x, offset_y, 0, 1]  # y軸は反転
        # ])

        #return projection * correction

        projection = matrix44.create_perspective_projection_from_bounds(
            (cx - width / 2 ) * near / fx,
            (cx + width / 2) * near / fx,
            (cy - height / 2) * near / fy,
            (cy + height / 2) * near / fy,
            near,
            far
        )
        return projection

    def parameters(self):
        """
        カメラパラメータの辞書形式の取得
        
        Returns:
        --------
        dict : カメラパラメータ
        """
        return {
            'K': self.K.tolist(),
            'R': self.R.tolist(),
            't': self.t.tolist()
        }

class Scene:
    gl_version = (3, 3, 0)
    def __init__(self, width=800, height=600):
        """
        3次元空間シミュレーション環境
        
        Parameters:
        -----------
        width : int
            画面の幅
        height : int
            画面の高さ
        """
        self.width = width
        self.height = height

        # GLFW init
        if not glfw.init():
            raise
        self.window = glfw.create_window(self.width, self.height, "Scene", None, None)
        if not self.window:
            glfw.terminate()
            raise
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()

        self.prog = self.ctx.program(
          vertex_shader='''
              #version 330
              
              uniform mat4 mvp;
              
              in vec3 in_position;
              in vec3 in_normal;
              
              out vec3 v_normal;
              out vec4 v_position;
              
              void main() {
                  gl_Position = mvp * vec4(in_position, 1.0);
                  
                  v_position = gl_Position;
                  v_normal = mat3(mvp) * in_normal;
              }
          ''',
          fragment_shader='''
              #version 330
              
              in vec3 v_normal;
              in vec4 v_position;
              
              out vec4 fragColor;
              
              // メタルマテリアルのパラメータ
              const vec3 metalColor = vec3(0.95, 0.93, 0.88);    // 金属の基本色
              const float roughness = 0.3;                        // 粗さ（小さいほど鏡面反射が強い）
              const float metallic = 0.9;                         // 金属度
              const float ambient = 0.2;                          // 環境光
              
              void main() {
                  vec3 N = normalize(v_normal);
                  
                  // 主光源の設定
                  vec3 lightPos = vec3(50.0, 50.0, 50.0);
                  vec3 L = normalize(lightPos - v_position.xyz);
                  
                  // 反射ベクトル
                  vec3 R = reflect(-L, N);

                  // 拡散反射
                  float diffuse = max(dot(N, L), 0.0);
                  
                  // メタリックワークフロー
                  vec3 specColor = mix(vec3(0.04), metalColor, metallic);
                  vec3 diffuseColor = mix(metalColor, vec3(0.0), metallic);
                  
                  // 最終的な色の計算
                  vec3 finalColor = 
                      diffuseColor * diffuse +
                      ambient * metalColor;
                  
                  // HDRトーンマッピング
                  finalColor = finalColor / (finalColor + vec3(1.0));
                  
                  // 深度情報をアルファチャンネルに格納
                  float depth = v_position.z / v_position.w;
                  fragColor = vec4(finalColor, depth);
              }
          '''
        )
        

        # フレームバッファの設定
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.width, self.height), 4)],
            depth_attachment=self.ctx.depth_texture((self.width, self.height))
        )
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)


    def load_stl(self, filename, z_offset=100, rotate=(0, 0, 0)):
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

        # Z方向に平行移動
        vertices[2::3] += -z_offset * 10.

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.nbo = self.ctx.buffer(normals.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '3f', 'in_position'),
                (self.nbo, '3f', 'in_normal'),
            ]
        )

    def render(self, camera: CameraParameters, model=None):
        """
        画像の描画

        Parameters:
        -----------
        camera : CameraParameters
            カメラパラメータ
        """
        
        # ビュー行列の作成
        # Z軸方向を見るようにカメラを設定
        view = np.eye(4)
        view[:3, :3] = camera.R.T
        view[:3, 3] = -camera.R.T @ camera.t.flatten()
        # MVP (Model-View-Projection) 行列の計算
        projection = camera.compute_projection_matrix(self.width, self.height)        
        if model is None:
            model = np.eye(4)
        mvp = projection @ view @ model
        
        self.prog['mvp'].write(mvp.astype('f4').tobytes())

        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render()
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()

        image = Image.frombytes("RGBA", self.fbo.size, self.fbo.read(components=4))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        # 深度情報はアルファチャンネルに格納されている
        depth_map = np.array(image)[:, :, 3]
        return image, depth_map

    def cleanup(self):
        self.vao.release()
        self.vbo.release()
        self.nbo.release()
        self.fbo.release()
        self.prog.release()
        self.ctx.release()
        glfw.terminate()

class StereoSystem:
    def __init__(self, left_camera: CameraParameters, right_camera: CameraParameters):
        """
        ステレオカメラシステム
        
        Parameters:
        -----------
        left_camera : CameraParameters
            左カメラのパラメータ
        right_camera : CameraParameters
            右カメラのパラメータ
        """
        self.left_cam = left_camera
        self.right_cam = right_camera

    def capture_stereo_images(self, scene: Scene, save_dir="out"):
        """
        ステレオ画像の撮影
        
        Parameters:
        -----------
        scene : Scene
            3次元シーン
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # 左カメラの画像を撮影
        image_left, depth_left = scene.render(self.left_cam)
        
        # 右カメラの画像を撮影
        image_right, depth_right = scene.render(self.right_cam)
        
        # 画像の保存
        image_left.save(save_dir / "left.png")
        image_right.save(save_dir / "right.png")
        
        # 深度マップの保存
        np.save(save_dir / "depth_left.npy", depth_left)
        np.save(save_dir / "depth_right.npy", depth_right)
        
        # カメラパラメータをyaml形式で保存
        left_cam_params = self.left_cam.parameters()
        right_cam_params = self.right_cam.parameters()
        with open(save_dir / "left_camera.yaml", "w") as f:
            yaml.dump(left_cam_params, f)
        with open(save_dir / "right_camera.yaml", "w") as f:
            yaml.dump(right_cam_params, f)
        
        # 対応点の生成と保存
        correspondences = self.generate_correspondences(depth_left, depth_right, scene.width, scene.height)
        with open(save_dir / "correspondences.yaml", "w") as f:
            yaml.dump(correspondences, f)
        
    def generate_correspondences(self, left_depth, right_depth, width, height):
        """
        深度マップから対応点を生成
        
        Parameters:
        -----------
        left_depth : ndarray
            左画像の深度マップ
        right_depth : ndarray
            右画像の深度マップ
        width : int
            画像の幅
        height : int
            画像の高さ

        Returns:
        --------
        dict : 対応点情報
        """
        correspondences = {
            'left_points': [],
            'right_points': [],
            'depths': []
        }
        
        # 画像全体からサンプリング
        for y in range(0, height, 10):  # 10ピクセルごとにサンプリング
            for x in range(0, width, 10):
                if left_depth[y, x] > 0:  # 有効な深度がある点のみ
                    depth = left_depth[y, x]
                    
                    # 3次元点の復元
                    focal_length = self.left_cam.K[1, 1]
                    X = (x - self.left_cam.K[2, 0]) * depth / focal_length
                    Y = (y - self.left_cam.K[2, 1]) * depth / focal_length
                    Z = depth
                    
                    # 右カメラでの投影
                    X_right = X - self.baseline  # ベースライン分のシフト
                    x_right = int(focal_length * X_right / Z + self.right_cam.K[2, 0])
                    y_right = int(focal_length * Y / Z + self.right_cam.K[2, 1])
                    
                    # 画像内かつ深度が一致する点のみ保存
                    if (0 <= x_right < self.width and 0 <= y_right < self.height and
                        abs(right_depth[y_right, x_right] - depth) < 0.01):
                        correspondences['left_points'].append([x, y])
                        correspondences['right_points'].append([x_right, y_right])
                        correspondences['depths'].append(depth)
        
        return correspondences
    
def reconstruct_3d_points(left_points, right_points, K, R_left, t_left, R_right, t_right):
    """
    対応点から3次元点群を復元
    
    Parameters:
    -----------
    left_points : list of [x, y]
        左画像の対応点
    right_points : list of [x, y]
        右画像の対応点
    K : ndarray, shape (3, 3)
        カメラ内部パラメータ
    R_left, R_right : ndarray, shape (3, 3)
        左右カメラの回転行列
    t_left, t_right : ndarray, shape (3, 1)
        左右カメラの並進ベクトル
        
    Returns:
    --------
    ndarray : 3次元点群
    """
    points_3d = []
    
    for (x_l, y_l), (x_r, y_r) in zip(left_points, right_points):
        # 正規化画像座標に変換
        x_l_norm = (x_l - K[0, 2]) / K[0, 0]
        y_l_norm = (y_l - K[1, 2]) / K[1, 1]
        x_r_norm = (x_r - K[0, 2]) / K[0, 0]
        y_r_norm = (y_r - K[1, 2]) / K[1, 1]
        
        # DLT法による三角測量
        A = np.zeros((4, 4))
        
        # 左カメラの拘束
        A[0] = x_l_norm * R_left[2] - R_left[0]
        A[1] = y_l_norm * R_left[2] - R_left[1]
        
        # 右カメラの拘束
        A[2] = x_r_norm * R_right[2] - R_right[0]
        A[3] = y_r_norm * R_right[2] - R_right[1]
        
        # SVDで解を求める
        _, _, Vh = np.linalg.svd(A)
        point_3d = Vh[-1, :3] / Vh[-1, 3]
        
        points_3d.append(point_3d)
    
    return np.array(points_3d)

# 使用例
if __name__ == "__main__":
    # シミュレータの初期化
    scene = Scene(width=800, height=600)
    
    # STLファイルの読み込み
    scene.load_stl("../../data/hole.stl", z_offset=100, rotate=(45, 0, 0))
    
    # カメラパラメータの設定
    baseline = 50
    left_cam = CameraParameters(fx=1000, fy=1000, cx=0, cy=0)
    right_cam = CameraParameters(fx=1000, fy=1000, cx=-baseline, cy=0, R=np.eye(3), t=np.array([[baseline], [0], [0]])) # ベースラインを50に設定
    system = StereoSystem(left_camera=left_cam, right_camera=right_cam)
    # ステレオ画像の撮影と保存
    save_dir = "out"
    system.capture_stereo_images(scene, save_dir=save_dir)
    
    print("Stereo images captured")
    
    # 保存されたデータの読み込みと3次元復元
    with open(f"{save_dir}/left_camera.yaml", "r") as f:
        left_params = yaml.load(f)
    with open(f"{save_dir}/right_camera.yaml", "r") as f:
        right_params = yaml.load(f)
    with open(f"{save_dir}/correspondences.yaml", "r") as f:
        corr = yaml.load(f)

    # 3次元復元
    points_3d = reconstruct_3d_points(
        corr['left_points'],
        corr['right_points'],
        np.array(left_params['K']),
        np.array(left_params['R']),
        np.array(left_params['t']),
        np.array(right_params['R']),
        np.array(right_params['t'])
    )
    
    print(f"Reconstructed {len(points_3d)} 3D points")