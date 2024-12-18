# mypy: ignore-errors

from pathlib import Path
import glm
import numpy as np
import yaml
from pyrr import Vector3, Matrix44

from ros_pc2i.base.mesh import load_stl
from ros_pc2i.base.renderer import Renderer
from ros_pc2i.base.camera import Camera
from ros_pc2i.base.scene import Scene
from ros_pc2i.materials.metal import Metal

class StereoSystem:
    def __init__(self, left_camera: Camera, right_camera: Camera):
        """
        ステレオカメラシステム
        
        Parameters:
        -----------
        left_camera : Camera
            左カメラのパラメータ
        right_camera : Camera
            右カメラのパラメータ
        """
        self.left_cam = left_camera
        self.right_cam = right_camera

    def capture_stereo_images(self, renderer: Renderer, scene: Scene, save_dir="out"):
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
        scene.light_position = Vector3([self.left_cam.t[0, 0], self.left_cam.t[1, 0], self.left_cam.t[2, 0]])
        image_left, depth_left = renderer.render(scene, self.left_cam)
        
        # 右カメラの画像を撮影
        scene.light_position = Vector3([self.right_cam.t[0, 0], self.right_cam.t[1, 0], self.right_cam.t[2, 0]])
        image_right, depth_right = renderer.render(scene, self.right_cam)
        
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
        
# 使用例
if __name__ == "__main__":
    width = 800
    height = 600
    
    renderer = Renderer(width, height, headless=True)
    scene = Scene()
    
    # STLファイルの読み込み
    mesh_path = Path(__file__).parent.parent / "data" / "hole.stl"
    mesh = load_stl(renderer.ctx, mesh_path)
    mesh.transform.position = Vector3([0, 0, 1000])
    mesh.transform.rotation = Vector3([np.pi/4, np.pi/4, np.pi/4]) #45度回転
    mesh.material = Metal()
    scene.add_mesh(mesh)
    
    # カメラパラメータの設定
    baseline = 50
    left_cam = Camera(fx=800, fy=600, cx=0, cy=0)
    right_cam = Camera(fx=800, fy=600, cx=-baseline, cy=0, R=np.eye(3), t=np.array([[baseline], [0], [0]])) # ベースラインを50に設定
    system = StereoSystem(left_camera=left_cam, right_camera=right_cam)
    # ステレオ画像の撮影と保存
    save_dir = "out"
    system.capture_stereo_images(renderer, scene, save_dir=save_dir)

    print("Stereo images captured")