import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 读取天气数据（4x7）
weather_data = np.loadtxt('weather_成都.txt')

# 读取平均天气数据（1x7）
mean_weather = np.loadtxt('weather_北京1_7取整.txt')

# 如果是 1D 向量，需要 reshape 成 2D 向量用于 sklearn
if mean_weather.ndim == 1:
    mean_weather = mean_weather.reshape(1, -1)

# 计算每一行与平均天气之间的余弦相似度（结果为 4x1）
cos_sim = cosine_similarity(weather_data, mean_weather)

# 归一化为权重（使和为1）
weights = cos_sim / np.sum(cos_sim)

# 保存结果为 4x1 文件
np.savetxt('C_成都_北京时间权重.txt', weights, fmt='%.8f')
