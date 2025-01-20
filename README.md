# ROS PointCloud to Image

- High quality mesh renderer with OpenGL
- 2D capturing image as like stereo camera
- WIP

## Install
```sh
pip install git+https://github.com/CA-Giken/ros-pc2i.git
```

## Usage
```python
from ros_pc2i import capture_image

STL_PATH = "data/hole.stl"

capture_image(mesh_path=STL_PATH)
```

## Reference

### `capture_image`

Capturing images with two eye stereo cameras.

#### Arguments

- mesh_path: Path (Required)
  STLファイルパス
- outdir : Path
  出力画像の保存先ディレクトリ
- width : int
  画像の幅(デフォルト800)
- height : int
  画像の高さ(デフォルト600)
- mesh_position : NDArray
  メッシュの位置(x, y, z)
  デフォルト: [0, 0, 200]
- mesh_rotation : NDArray
  メッシュの回転角度(x, y, z)
  デフォルト: [np.pi/8, 0, np.pi/4]
- mesh_material : Material
  メッシュのマテリアル
  デフォルト: Metal()
- cam_baseline : int
  カメラのベースライン(視差)[mm]
  デフォルト: 50mm