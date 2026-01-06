"""
查询 Tushare 用户积分和到期时间
"""
import tushare as ts

# Tushare API KEY
TUSHARE_TOKEN = "0c1ee6d6473ee20d85144b8fd4f8f5cf6a3fd0d505fc09b029546134"

def query_user_info():
    """查询用户积分和到期时间"""
    try:
        # 初始化 API
        print("正在初始化 API...")
        pro = ts.pro_api(TUSHARE_TOKEN)
        print("✓ API 初始化成功\n")
        
        # 查询用户信息
        print("正在查询用户信息...")
        # 需要传入 token 参数
        df = pro.user(token=TUSHARE_TOKEN)
        
        print(f"查询结果类型: {type(df)}")
        print(f"DataFrame 长度: {len(df)}\n")
        
        if df is not None and len(df) > 0:
            print("=" * 80)
            print("Tushare 用户信息")
            print("=" * 80)
            print(df.to_string(index=False))
            print("=" * 80)
            
            # 解析关键信息
            print("\n关键信息：")
            for col in df.columns:
                value = df[col].values[0]
                print(f"  {col}: {value}")
            
            print("\n✅ 查询成功！")
            return True
        else:
            print("⚠️  返回结果为空\n可能原因：")
            print("  1. API KEY 权限不足")
            print("  2. 积分信息不可用")
            return False
            
    except Exception as e:
        print(f"\n✗ 查询失败: {e}")
        import traceback
        print("\n详细错误:")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    query_user_info()
