import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial import distance
from scipy.stats import entropy

# 读取数据
data = pd.read_csv('S_2W_北京市station_换电数据_flow_2_del_normalized.csv')

# 获取所有station
stations = data['name'].unique()

# 存储指标结果
result = []

# 自定义计算JSD（连续数据离散化）
def calculate_jsd(real, pred, bins=20):
    real_hist, _ = np.histogram(real, bins=bins, range=(min(real.min(), pred.min()), max(real.max(), pred.max())), density=True)
    pred_hist, _ = np.histogram(pred, bins=bins, range=(min(real.min(), pred.min()), max(real.max(), pred.max())), density=True)

    # 防止出现0
    real_prob = real_hist + 1e-8
    pred_prob = pred_hist + 1e-8

    real_prob /= real_prob.sum()
    pred_prob /= pred_prob.sum()

    m = 0.5 * (real_prob + pred_prob)
    jsd = 0.5 * (entropy(real_prob, m) + entropy(pred_prob, m))
    return jsd

for station in stations:
    station_data = data[data['name'] == station]
    real = station_data['Normalized Data'].values
    pred = station_data['average_value'].values
    variance = station_data['variance'].iloc[0]  # 直接提取 'variance' 列的值

    # MAE
    mae = mean_absolute_error(real, pred)

    # RMSE
    rmse = np.sqrt(mean_squared_error(real, pred))

    # MAPE，避免除0
    mape = np.mean(np.abs((real - pred) / (real + 1e-8)))

    # MMD
    mmd = np.mean((np.mean(real) - np.mean(pred)) ** 2)

    # 改进后的JSD
    jsd = calculate_jsd(real, pred, bins=20)

    result.append({
        'station': station,
        'MAE Value': mae,
        'RMSE Value': rmse,
        'MAPE Value': mape,
        'MMD Value': mmd,
        'JSD Value': jsd,
        'variance': variance  # 直接添加 variance 列
    })

# 保存结果
result_df = pd.DataFrame(result)
result_df.to_csv('S_2W_北京市station_指标评估结果.csv', index=False, encoding='utf-8-sig')

print('评估完成，结果已保存至 S_2W_北京市station_指标评估结果.csv')
