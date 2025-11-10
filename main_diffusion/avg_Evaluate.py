import pandas as pd

# 读取原始数据
df = pd.read_csv('2W_北京市station_换电数据_flow_2_del_normalized.csv')

# 按name分组，对flow、Normalized Data 和 average_value 求均值
result = df.groupby('name', as_index=False)[['flow', 'Normalized Data', 'average_value', 'variance']].mean()

# 保存结果
result.to_csv('S_2W_北京市station_换电数据_flow_2_del_normalized.csv', index=False)

print("处理完成，结果已保存为 S_2W_北京市station_换电数据_flow_2_del_normalized.csv")
