"""
测试 Tushare API KEY 是否有效
"""
import sys

# Tushare API KEY
TUSHARE_TOKEN = "0c1ee6d6473ee20d85144b8fd4f8f5cf6a3fd0d505fc09b029546134"

def test_tushare():
    """测试 Tushare 连接"""
    try:
        import tushare as ts
        print("✓ Tushare 模块导入成功")
        
        # 初始化 Tushare Pro
        pro = ts.pro_api(TUSHARE_TOKEN)
        print("✓ Tushare 连接成功")
        
        # 测试获取股票列表（获取最近的股票基础信息）
        print("\n正在测试数据获取...")
        df = pro.stock_basic(exchange='', list_status='L')
        
        if df is not None and not df.empty:
            print(f"✓ 成功获取数据，共 {len(df)} 条记录")
            print("\n前5条数据:")
            print(df.head())
            print("\n✅ API KEY 有效！可以正常使用")
            return True
        else:
            print("✗ 获取数据为空")
            return False
            
    except ImportError:
        print("✗ Tushare 模块未安装")
        print("  请运行: pip install tushare")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        print("\n可能原因:")
        print("1. API KEY 无效或已过期")
        print("2. 网络连接问题")
        print("3. Tushare 服务异常")
        return False

if __name__ == '__main__':
    success = test_tushare()
    sys.exit(0 if success else 1)
