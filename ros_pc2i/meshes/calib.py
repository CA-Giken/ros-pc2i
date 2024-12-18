from ros_pc2i.lib.stereo import Scene

class CalibrationBoard:
    def __init__(self, scene=Scene, width=100, height=100, squares=10):
        """
        チェッカーボード描画クラス
        
        Parameters
        ----------
        width : int
            チェッカーボードの幅(mm)
        height : int
            チェッカーボードの高さ(mm)
        squares : int
            チェッカーボードの1辺のマス数
        """
        self.width = width
        self.height = height
        self.squares = squares
        
        texture_size = 512
        checker_img = 