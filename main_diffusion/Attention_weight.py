import torch
import torch.nn.functional as F

# 读取txt为Tensor
def load_kg(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        # 按空格或制表符分割
        items = list(map(float, line.strip().split()))
        data.append(items)
    return torch.tensor(data, dtype=torch.float)

# 数据路径
chengdu_file = '成都知识图谱.txt'
beijing_file = '北京知识图谱1_4.txt'

# 加载数据
chengdu = load_kg(chengdu_file)  # [24, 4]
beijing = load_kg(beijing_file)  # [1, 4]

print("成都数据 shape:", chengdu.shape)
print("北京数据 shape:", beijing.shape)

# 注意力机制计算权重
scores = torch.matmul(chengdu, beijing.T)  # [24, 1]
attn_weights = F.softmax(scores, dim=0)    # 归一化 [24, 1]

print("成都每个区域的注意力权重：")
print(attn_weights.squeeze())


# 保存成txt，只保存数字
save_path = '54_A_成都_北京区域权重.txt'

with open(save_path, 'w', encoding='utf-8') as f:
    for weight in attn_weights.squeeze():
        f.write(f'{weight.item():.6f}\n')

print(f'权重已保存到: {save_path}')