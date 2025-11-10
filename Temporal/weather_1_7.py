import numpy as np

# 读取原始 4x7 的文件（默认为空格或制表符分隔）
data = np.loadtxt('weather_北京.txt')

# 对每一列求平均并四舍五入取整
col_mean_int = np.rint(np.mean(data, axis=0)).astype(int)

# 保存为 1x7 的文本文件（使用制表符分隔）
np.savetxt('weather_北京1_7取整.txt', [col_mean_int], fmt='%d', delimiter='\t')
