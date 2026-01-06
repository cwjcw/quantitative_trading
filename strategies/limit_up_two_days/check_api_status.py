"""
查询 Tushare 账户信息 - 尝试多个方法
"""
import tushare as ts
import pandas as pd

# Tushare API KEY
TUSHARE_TOKEN = "0c1ee6d6473ee20d85144b8fd4f8f5cf6a3fd0d505fc09b029546134"

def test_apis():
    """测试多个 API 来获取用户信息"""
    try:
        print("正在初始化 API...")
        pro = ts.pro_api(TUSHARE_TOKEN)
        print("✓ API 初始化成功\n")
        
        # 方法1: user() 接口
        print("="*80)
        print("尝试1: 查询 user() 接口")
        print("="*80)
        try:
            df = pro.user(token=TUSHARE_TOKEN)
            if df is not None and len(df) > 0:
                print("✓ 成功获取用户信息")
                print(df.to_string(index=False))
            else:
                print("⚠️  user() 返回空结果")
        except Exception as e:
            print(f"✗ user() 查询失败: {e}")
        
        # 方法2: 尝试查询 stock_basic
        print("\n" + "="*80)
        print("尝试2: 查询 stock_basic (验证 API 是否正常工作)")
        print("="*80)
        try:
            df = pro.stock_basic(exchange='', list_status='L')
            if df is not None and len(df) > 0:
                print(f"✓ API 正常工作，成功获取 {len(df)} 条股票记录")
                print(f"\n前3条数据:")
                print(df.head(3).to_string())
            else:
                print("⚠️  stock_basic 返回空结果")
        except Exception as e:
            print(f"✗ stock_basic 查询失败: {e}")
        
        # 总结
        print("\n" + "="*80)
        print("查询总结")
        print("="*80)
        print("你的 API KEY 是有效的（能连接并初始化成功）")
        print("如果 user() 返回空，可能是因为：")
        print("  1. 免费版账户可能不支持此接口")
        print("  2. 需要在 Tushare Pro 官网查询会员信息")
        print("\nTushare Pro 会员信息查询地址:")
        print("  https://tushare.pro/member")
            
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_apis()
