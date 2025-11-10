import pandas as pd

# 加载数据
df = pd.read_csv('C_成都_北京generated_samples.csv')

# 计算 sample_1 到 sample_10 的方差
samples_columns = [f'sample_{i}' for i in range(1, 11)]

# 添加 'variance' 列，计算每行样本列的方差
df['variance'] = df[samples_columns].var(axis=1)

# 保存更新后的 DataFrame 到一个新的 CSV 文件
df.to_csv('variance_C_成都_北京generated_samples.csv', index=False)

print("文件已保存为 'variance_C_成都_北京generated_samples.csv'")
