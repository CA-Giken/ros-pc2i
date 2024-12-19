# mypy: ignore-errors

import numpy as np
from pyrr import matrix44
from numpy.typing import NDArray

class Camera:
    def __init__(self, fx, fy, cx, cy, R=np.eye(3), t=np.zeros((3, 1))):
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
        self.R = R
        self.t = t
        
        self._view_matrix = self._create_look_at(
            position=np.array([0, 0, 0]),
            target=np.array([0, 0, 1]),
            up=np.array([0, 1, 0])
        )
        
    def compute_projection_matrix(self, width, height, near=1, far=1000):
        """
        射影行列の計算
        
        Returns:
        --------

        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

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
        return {
            'K': self.K.tolist(),
            'R': self.R.tolist(),
            't': self.t.tolist()

        }
    
    def _create_look_at(
        self,
        position: NDArray[np.float32],
        target: NDArray[np.float32],
        up: NDArray[np.float32]
    ):
        """
        カメラのビュー行列を計算
        
        Parameters:
        -----------
        position : ndarray, shape (3,)
            カメラの位置
        target : ndarray, shape (3,)
            カメラの注視点
        up : ndarray, shape (3,)
            カメラの上方向
        """
        return matrix44.create_look_at(
            position,
            target,
            up,
            dtype=np.float32
        )

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
