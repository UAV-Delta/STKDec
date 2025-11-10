import setproctitle

import numpy as np
import json
import os
from tqdm import trange
import torch

import torch.nn as nn
from torch_geometric.data import Data


#######################################图结构数据#############################################################
# 加载站点向量表示
station_features = np.load(r'id2emb.npy')

# 加载站点边索引和权重
with open(r'entity2idx.json', 'r') as f:
    entity2idx = json.load(f)

# 先过滤出前缀为 'station' 的键，然后对它们进行排序
sorted_station_keys = sorted([key for key in entity2idx.keys() if key.startswith("beijing_station")], key=lambda x: int(x.replace('beijing_station', '')))


# 根据排序后的键生成 station_indices 字典
station_indices = {key: entity2idx[key]['idx'] for key in sorted_station_keys}
#print('station的顺序',station_indices)

# 加载站点的距离矩阵

# 确保站点数量与距离矩阵一致
num_stations = len(station_indices)

# 创建 PyTorch Geometric 数据对象
station_vectors = []
edge_index = []
edge_attr = []

for i, (station_name, idx) in enumerate(station_indices.items()):
    # 获取站点向量表示
    vector = station_features[idx]
    station_vectors.append(vector)


#print('点的顺序',station_vectors)

# 将列表转换为 numpy 数组
station_vectors = np.array(station_vectors)
edge_index = np.array(edge_index).T  # 转置以符合 PyTorch Geometric 的格式
edge_attr = np.array(edge_attr)

np.savetxt('北京知识图谱.txt', station_vectors)
# 创建 PyTorch Geometric 数据对象
#station_vectors是点,edge_index是边,edge_attr是权重，是一个有向图。
#您有 54 个站点（节点），它们的特征向量维度是 4，共有 2862 条边及其对应的权重。

#print(station_features.shape)