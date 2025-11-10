import numpy as np

# 输入输出路径
input_path = 'KL_成都_北京时间权重.txt'
output_path = 'KL_成都_北京时间权重_Max1.txt'

# 读取权重
weights = np.loadtxt(input_path)  # shape (24,)

# 找最大值
max_weight = np.max(weights)

# 按比例缩放
new_weights = weights / max_weight  # 最大值为1，其他等比例缩放

# 保存新权重
np.savetxt(output_path, new_weights, fmt='%.6f')

print(f'最大值已缩放为1，结果保存至 {output_path}，shape: {new_weights.shape}')
