import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 读取文件
def read_txt(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append([float(x) for x in line.strip().split()])
    return np.array(data)

# 路径
chengdu_path = '成都知识图谱2750.txt'
beijing_path = '北京知识图谱1_4.txt'
output_path = 'cos成都_北京区域权重.txt'

# 读取数据
chengdu_kg = read_txt(chengdu_path)  # (24, 4)
beijing_kg = read_txt(beijing_path)  # (1, 4)

# 计算余弦相似度 (24, 1)
sim = cosine_similarity(chengdu_kg, beijing_kg)

# 归一化为权重 (和为1)
weights = sim / np.sum(sim)

# 保存结果
np.savetxt(output_path, weights, fmt='%.6f')

print(f'余弦权重保存成功，shape: {weights.shape}')
