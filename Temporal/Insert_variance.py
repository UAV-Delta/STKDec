import pandas as pd

# ===== 第一步：计算每个 station 的方差并保存 =====

# 加载生成样本数据
df_samples = pd.read_csv('Time_W_成都_北京generated_samples.csv')

# 计算每行样本的方差
sample_columns = [f'sample_{i}' for i in range(1, 11)]
df_samples['variance'] = df_samples[sample_columns].var(axis=1)

# 加载真实数据（包含 name 列）
df_real = pd.read_csv('Time_W_北京市station_换电数据_flow_2_del_normalized.csv')

# 添加方差列
df_real['variance'] = df_samples['variance']

# 按 name 分组求平均方差
df_grouped = df_real.groupby('name')['variance'].mean().reset_index()

# 排序：station1-station24
df_grouped['station_num'] = df_grouped['name'].str.extract('station(\d+)', expand=False).astype(int)
df_grouped = df_grouped.sort_values('station_num').drop(columns='station_num')

# 保存为中间文件（可省略）
df_grouped.to_csv('variance_Time_W_成都_北京.csv', index=False)

# ===== 第二步：将方差列添加到 station 评估表格中 =====

# 加载评估表格
# 读取评估表格
df_eval = pd.read_csv('Time_W_成都_北京station评估.csv')
df_grouped = pd.read_csv('variance_Time_W_成都_北京.csv')

# 清洗列名（防止空格或隐藏字符）
df_eval.columns = df_eval.columns.str.strip()
df_grouped.columns = df_grouped.columns.str.strip()

# 合并：将 df_grouped 的 'variance' 按照 name 对应插入到 df_eval 的 'station'
df_merged = pd.merge(df_eval, df_grouped, left_on='station', right_on='name', how='left')

# 删除多余的 'name' 列（因为已经有 station 了）
df_merged = df_merged.drop(columns=['name'])

# 保存结果
df_merged.to_csv('variance_Time_W_成都_北京station评估.csv', index=False)
