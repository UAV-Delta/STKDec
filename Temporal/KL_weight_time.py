import numpy as np
from scipy.special import rel_entr  # 用于计算KL散度

# ===== 读取数据 =====
# 成都天气：4行7列，每行表示一周
weather_cd = np.loadtxt('weather_成都.txt')  # shape: (4, 7)

# 北京平均天气：1行7列
weather_bj = np.loadtxt('weather_北京1_7取整.txt')  # shape: (7,) or (1, 7)

# 如果是 1D 向量，reshape 成 1x7
if weather_bj.ndim == 1:
    weather_bj = weather_bj.reshape(1, -1)

# 将天气数据归一化为概率分布（和为1）
def normalize_to_distribution(arr):
    arr = np.maximum(arr, 1e-10)  # 避免除0或log(0)
    return arr / np.sum(arr, axis=1, keepdims=True)

weather_cd_prob = normalize_to_distribution(weather_cd)  # shape: (4, 7)
weather_bj_prob = normalize_to_distribution(weather_bj)  # shape: (1, 7)

# ===== 计算每一行与北京天气的 KL 散度 =====
# KL(P||Q): P = 成都周天气，Q = 北京平均天气
kl_divergences = []
for i in range(weather_cd_prob.shape[0]):
    kl = np.sum(rel_entr(weather_cd_prob[i], weather_bj_prob[0]))
    kl_divergences.append(kl)

kl_divergences = np.array(kl_divergences)  # shape: (4,)

# ===== 转换为相似度（值越小代表越相似） -> 权重（值越大代表越重要） =====
# 使用 softmax(-KL) 来获得归一化权重
similarities = np.exp(-kl_divergences)
weights = similarities / np.sum(similarities)

# ===== 保存为 4x1 的文件 =====
np.savetxt('test_KL_成都_北京时间权重.txt', weights.reshape(-1, 1), fmt='%.8f')

print("KL权重计算完成，已保存为 'test_KL_成都_北京时间权重.txt'")

print("KL散度:", kl_divergences)
print("exp(-KL):", np.exp(-kl_divergences))
print("Softmax权重:", weights)
