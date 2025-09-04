#!/usr/bin/python
# -*-coding:utf-8-*-
"""
測試 MockTile 是否正確添加到 maxcut_main.py
"""

def test_mock_tile_import():
    """測試 MockTile 導入"""
    print("🧪 測試 MockTile 導入")
    print("=" * 50)
    
    try:
        # 嘗試導入 MockTile
        from maxcut_main import MockTile
        print("✅ MockTile 導入成功")
        
        # 創建實例
        tile = MockTile()
        print("✅ MockTile 實例創建成功")
        
        # 測試方法
        area = tile.calculate_tile_area()
        power = tile.calculate_tile_power()
        print(f"✅ 面積計算: {area} um²")
        print(f"✅ 功率計算: {power} W")
        
        return True
        
    except ImportError as e:
        print(f"❌ MockTile 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_maxcut_tcg_tile():
    """測試 MaxCutTCG 的 tile 屬性"""
    print("\n🧪 測試 MaxCutTCG 的 tile 屬性")
    print("=" * 50)
    
    try:
        from maxcut_main import MaxCutTCG
        
        # 創建一個簡單的結構
        structure = {
            'layers': [
                {'input_size': 3, 'output_size': 3}
            ]
        }
        
        # 創建 MaxCutTCG 實例
        tcg = MaxCutTCG(structure, "SimConfig.ini")
        print("✅ MaxCutTCG 創建成功")
        
        # 測試 tile 屬性
        tile = tcg.tile
        print("✅ tile 屬性訪問成功")
        
        # 測試 tile 方法
        area = tile.calculate_tile_area()
        power = tile.calculate_tile_power()
        print(f"✅ tile 面積計算: {area} um²")
        print(f"✅ tile 功率計算: {power} W")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函式"""
    print("🚀 MockTile 修復驗證測試")
    print("=" * 80)
    
    # 測試導入
    success1 = test_mock_tile_import()
    
    if success1:
        # 測試 MaxCutTCG 的 tile 屬性
        success2 = test_maxcut_tcg_tile()
        
        if success2:
            print("\n" + "=" * 60)
            print("🎉 所有測試通過！")
            print("=" * 60)
            print("💡 現在可以嘗試完整的硬體建模:")
            print("python maxcut_main.py --graph_file test_graphs/defined_3nodes.txt --algorithm rram_psa")
        else:
            print("\n❌ MaxCutTCG tile 屬性測試失敗")
    else:
        print("\n❌ MockTile 導入測試失敗")

if __name__ == "__main__":
    main()
