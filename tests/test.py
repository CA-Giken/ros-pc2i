#!/usr/bin/env python
# -*- coding: utf-8 -*-
#mypy: ignore-errors

from ros_pc2i.pc2i import STLRenderer, CameraController, render_stl_to_image
import unittest
import os
import numpy as np
from PIL import Image
import glm
from stl import mesh

class TestSTLRenderer(unittest.TestCase):
    def setUp(self):
        """テスト用のシンプルな立方体STLファイルを生成"""
        # 立方体の頂点を定義
        vertices = np.array([
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ])
        
        # 面を定義
        faces = np.array([
            [0,3,1], [1,3,2],  # 前面
            [5,6,4], [4,6,7],  # 後面
            [0,1,5], [0,5,4],  # 下面
            [3,7,2], [2,7,6],  # 上面
            [1,2,5], [2,6,5],  # 右面
            [0,4,3], [3,4,7],  # 左面
        ])
        
        # メッシュを作成
        cube = mesh.Mesh(np.zeros(12, dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = vertices[f[j]]
        
        # テスト用STLファイルを保存
        self.test_dir = '../data'
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_stl_path = os.path.join(self.test_dir, 'test_cube.stl')
        cube.save(self.test_stl_path)
        
        # レンダラーを初期化
        self.renderer = STLRenderer(width=400, height=300)
        self.camera_pos = (5, 5, 5)
        
        # 初期化フラグ
        self.stl_loaded = False
        self.vao_created = False
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        try:
            if hasattr(self, 'renderer'):
                if hasattr(self.renderer, 'vao') and self.renderer.vao:
                    self.renderer.vao.release()
                if hasattr(self.renderer, 'vbo') and self.renderer.vbo:
                    self.renderer.vbo.release()
                if hasattr(self.renderer, 'nbo') and self.renderer.nbo:
                    self.renderer.nbo.release()
                if hasattr(self.renderer, 'ibo') and self.renderer.ibo:
                    self.renderer.ibo.release()
                if hasattr(self.renderer, 'prog') and self.renderer.prog:
                    self.renderer.prog.release()
                if hasattr(self.renderer, 'fbo') and self.renderer.fbo:
                    self.renderer.fbo.release()
                if hasattr(self.renderer, 'ctx') and self.renderer.ctx:
                    self.renderer.ctx.release()
        except Exception as e:
            print(f"クリーンアップ中にエラーが発生: {str(e)}")
        
        # テストファイルの削除
        try:
            if os.path.exists(self.test_stl_path):
                os.remove(self.test_stl_path)
            if os.path.exists(self.test_dir):
                os.rmdir(self.test_dir)
        except Exception as e:
            print(f"テストファイルの削除中にエラーが発生: {str(e)}")

    def test_stl_loading(self):
        """STLファイルの読み込みをテスト"""
        try:
            self.assertTrue(os.path.exists(self.test_stl_path), "テストSTLファイルが存在しません")
            num_indices = self.renderer.load_stl(self.test_stl_path)
            self.assertEqual(num_indices, 36)  # 12面 × 3頂点
            self.stl_loaded = True
            print("✓ STLファイルの読み込みに成功")
        except Exception as e:
            self.fail(f"STLファイルの読み込みに失敗: {str(e)}")

    def test_shader_compilation(self):
        """シェーダーのコンパイルをテスト"""
        try:
            self.assertIsNotNone(self.renderer.prog, "シェーダープログラムがNoneです")
            # 必要なユニフォーム変数の存在確認
            uniforms = self.renderer.prog._members
            required_uniforms = ['model', 'view', 'projection', 'viewPos']
            for uniform in required_uniforms:
                self.assertIn(uniform, uniforms, f"必要なuniform '{uniform}'が見つかりません")
            print("✓ シェーダーのコンパイルに成功")
            
            # デバッグ情報の出力
            print("\nシェーダーユニフォーム変数:")
            for name in uniforms:
                print(f"- {name}")
                
        except Exception as e:
            self.fail(f"シェーダーのコンパイルに失敗: {str(e)}")

    def test_buffer_creation(self):
        """バッファオブジェクトの作成をテスト"""
        try:
            self.renderer.load_stl(self.test_stl_path)
            
            # バッファの存在確認
            self.assertIsNotNone(self.renderer.vbo, "VBOがNoneです")
            self.assertIsNotNone(self.renderer.nbo, "NBOがNoneです")
            self.assertIsNotNone(self.renderer.ibo, "IBOがNoneです")
            self.assertIsNotNone(self.renderer.vao, "VAOがNoneです")
            
            # バッファサイズの確認
            vbo_size = self.renderer.vbo.size
            nbo_size = self.renderer.nbo.size
            ibo_size = self.renderer.ibo.size
            
            print(f"\nバッファサイズ:")
            print(f"VBO: {vbo_size} bytes")
            print(f"NBO: {nbo_size} bytes")
            print(f"IBO: {ibo_size} bytes")
            
            self.assertGreater(vbo_size, 0, "VBOサイズが0です")
            self.assertGreater(nbo_size, 0, "NBOサイズが0です")
            self.assertGreater(ibo_size, 0, "IBOサイズが0です")
            
            self.vao_created = True
            print("✓ バッファオブジェクトの作成に成功")
        except Exception as e:
            self.fail(f"バッファオブジェクトの作成に失敗: {str(e)}")

    def test_render_output(self):
        """レンダリング出力をテスト"""
        try:
            self.renderer.load_stl(self.test_stl_path)
            pixels = self.renderer.render(
                camera_pos=self.camera_pos,
                target_pos=(0, 0, 0),
                material_props={
                    'ambient': (0.2, 0.2, 0.2),
                    'diffuse': (0.8, 0.8, 0.8),
                    'specular': (1.0, 1.0, 1.0),
                    'shininess': 32.0
                },
                light_props={
                    'position': (10.0, 10.0, 10.0),
                    'ambient': (0.3, 0.3, 0.3),
                    'diffuse': (1.0, 1.0, 1.0),
                    'specular': (1.0, 1.0, 1.0)
                }
            )
            
            # ピクセルデータの検証
            image = Image.frombytes('RGBA', (400, 300), pixels)
            pixel_data = np.array(image)
            
            # 統計情報の収集
            pixel_stats = {
                'min': pixel_data.min(),
                'max': pixel_data.max(),
                'mean': pixel_data.mean(),
                'non_zero': np.count_nonzero(pixel_data)
            }
            
            # 画像の保存（デバッグ用）
            debug_dir = 'debug_output'
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, 'test_render_output.png')
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save(debug_path)
            
            # 検証
            self.assertGreater(pixel_stats['max'], 0, "画像が完全に黒です")
            self.assertGreater(pixel_stats['non_zero'], 1000, "非ゼロピクセルが少なすぎます")
            
            print("\nレンダリング統計:")
            print(f"最小値: {pixel_stats['min']}")
            print(f"最大値: {pixel_stats['max']}")
            print(f"平均値: {pixel_stats['mean']:.2f}")
            print(f"非ゼロピクセル数: {pixel_stats['non_zero']}")
            print(f"デバッグ画像保存先: {debug_path}")
            
            print("✓ レンダリング出力に成功")
            
        except Exception as e:
            self.fail(f"レンダリング出力のテストに失敗: {str(e)}")

    def test_camera_matrix(self):
        """カメラ行列の計算をテスト"""
        try:
            camera = CameraController(self.camera_pos)
            view_matrix = camera.get_view_matrix()
            
            # 行列の基本的な性質をチェック
            self.assertEqual(len(view_matrix), 4, "ビュー行列の行数が不正です")
            self.assertEqual(len(view_matrix[0]), 4, "ビュー行列の列数が不正です")
            
            # カメラの位置が正しく変換されることをテスト
            camera_pos_vec = glm.vec4(*self.camera_pos, 1.0)
            transformed = view_matrix * camera_pos_vec
            
            print("\nカメラ行列情報:")
            print(f"ビュー行列:\n{view_matrix}")
            print(f"カメラ位置: {self.camera_pos}")
            print(f"変換後の位置: {transformed}")
            
            # 変換後の位置が原点に近いことを確認
            self.assertAlmostEqual(transformed.x, 0.0, places=5)
            self.assertAlmostEqual(transformed.y, 0.0, places=5)
            
            print("✓ カメラ行列の計算に成功")
            
        except Exception as e:
            self.fail(f"カメラ行列のテストに失敗: {str(e)}")

def run_debug_renders():
    """デバッグ用のレンダリングテスト"""
    debug_dir = 'debug_output'
    os.makedirs(debug_dir, exist_ok=True)
    
    test_cases = [
        {
            'name': 'default',
            'camera_pos': (5, 5, 5),
            'material': None,
            'light': None
        },
        {
            'name': 'close_up',
            'camera_pos': (2, 2, 2),
            'material': {
                'ambient': (0.3, 0.3, 0.3),
                'diffuse': (0.8, 0.8, 0.8),
                'specular': (1.0, 1.0, 1.0),
                'shininess': 64.0
            },
            'light': {
                'position': (5.0, 5.0, 5.0),
                'ambient': (0.3, 0.3, 0.3),
                'diffuse': (1.0, 1.0, 1.0),
                'specular': (1.0, 1.0, 1.0)
            }
        },
        {
            'name': 'side_view',
            'camera_pos': (10, 0, 0),
            'material': {
                'ambient': (0.2, 0.2, 0.2),
                'diffuse': (0.7, 0.7, 0.7),
                'specular': (1.0, 1.0, 1.0),
                'shininess': 32.0
            },
            'light': {
                'position': (10.0, 10.0, 10.0),
                'ambient': (0.2, 0.2, 0.2),
                'diffuse': (0.8, 0.8, 0.8),
                'specular': (1.0, 1.0, 1.0)
            }
        }
    ]
    
    test_stl_path = os.path.join('test_data', 'test_cube.stl')
    
    for case in test_cases:
        print(f"\nテストケース: {case['name']}")
        try:
            output_path = os.path.join(debug_dir, f'debug_render_{case["name"]}.png')
            render_stl_to_image(
                test_stl_path,
                camera_pos=case['camera_pos'],
                output_path=output_path,
                material_props=case['material'],
                light_props=case['light']
            )
            print(f"✓ {case['name']} レンダリングに成功: {output_path}")
        except Exception as e:
            print(f"✗ {case['name']} レンダリングに失敗: {str(e)}")

if __name__ == '__main__':
    # 単体テストを実行
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # デバッグレンダリングを実行
    print("\n=== デバッグレンダリングの実行 ===")
    run_debug_renders()
