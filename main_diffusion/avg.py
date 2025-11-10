# 读取1.txt，计算每列均值，写入2.txt

input_file = '北京知识图谱.txt'   # 输入文件路径
output_file = '北京知识图谱1_4.txt'  # 输出文件路径

import numpy as np

# 读取数据
data = []
with open(input_file, 'r') as f:
    for line in f:
        # 按空格分割，每行转为float
        row = [float(x) for x in line.strip().split()]
        data.append(row)

data = np.array(data)  # 转为numpy数组，shape: (24, 4)

# 计算每列均值
mean_vals = np.mean(data, axis=0)  # shape: (4,)

# 保存到2.txt
with open(output_file, 'w') as f:
    f.write(' '.join([str(x) for x in mean_vals.tolist()]))

print("处理完成，结果已保存到", output_file)
