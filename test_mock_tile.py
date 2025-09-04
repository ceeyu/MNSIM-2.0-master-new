#!/usr/bin/python
# -*-coding:utf-8-*-
"""
測試 MockTile 類別
"""

from maxcut_main import MockTile

def test_mock_tile():
    """測試 MockTile 類別"""
    print("🧪 測試 MockTile 類別")
    print("=" * 50)
    
    try:
        # 創建 MockTile 實例
        tile = MockTile()
        print("✅ MockTile 創建成功")
        
        # 測試面積計算
        area = tile.calculate_tile_area()
        print(f"✅ 面積計算成功: {area} um²")
        
        # 測試功率計算
        power = tile.calculate_tile_power()
        print(f"✅ 功率計算成功: {power} W")
        
        print("\n🎉 MockTile 測試通過！")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mock_tile()
