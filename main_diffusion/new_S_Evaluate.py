import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial.distance import jensenshannon

# ===================== 读取数据 =======================
df = pd.read_csv(r'new_1W_北京市station_换电数据_flow_2_del_normalized.csv')
df['hour'] = pd.to_datetime(df['hour'])
df['date'] = df['hour'].dt.date
df['hour_of_day'] = df['hour'].dt.hour

stations = df['name'].unique()[:24]
dates = df['date'].unique()
max_val = df['flow'].max()
min_val = df['flow'].min()

# ===================== 工具函数 =======================
def nth_largest(array, n):
    flattened = array.flatten()
    sorted_array = np.sort(flattened)
    return sorted_array[-n]

def nth_smallest(array, n):
    flattened = array.flatten()
    sorted_array = np.sort(flattened)
    return sorted_array[n - 1]

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def calculate_jsd(matrix1, matrix2):
    data1 = matrix1.flatten()
    data2 = matrix2.flatten()
    min_val = min(data1.min(), data2.min())
    max_val = max(data1.max(), data2.max())
    bins = np.linspace(min_val, max_val, 100)
    hist1, _ = np.histogram(data1, bins=bins, density=True)
    hist2, _ = np.histogram(data2, bins=bins, density=True)
    hist1 = np.where(hist1 == 0, 1e-10, hist1)
    hist2 = np.where(hist2 == 0, 1e-10, hist2)
    return jensenshannon(hist1, hist2) ** 2

# ===================== 稳定版MMD计算 =======================
def gaussian_kernel_matrix(x, y, sigma=50.0):
    beta = 1.0 / (2.0 * sigma ** 2)
    dist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    return np.exp(-beta * dist)

def calculate_mmd_stable(x, y, sigma=50.0):
    Kxx = gaussian_kernel_matrix(x, x, sigma)
    Kyy = gaussian_kernel_matrix(y, y, sigma)
    Kxy = gaussian_kernel_matrix(x, y, sigma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

# ===================== 开始评估 =======================
results = []

for station in stations:
    df_station = df[df['name'] == station].copy()
    station_metrics = {'station': station, 'MAE': [], 'RMSE': [], 'MAPE': [], 'JSD': []}
    real_list = []
    pred_list = []

    for date in dates:
        daily_data = df_station[df_station['date'] == date].sort_values(by='hour_of_day')
        if len(daily_data) < 24:
            continue

        # 反归一化
        real = daily_data['Normalized Data'].values[:24] * (max_val - min_val) + min_val
        pred = daily_data['average_value'].values[:24] * (max_val - min_val) + min_val

        # 区间映射
        pred_min = pred.min()
        pred_max = pred.max()
        real_min = nth_smallest(real, 5)
        real_max = nth_largest(real, 5)
        pred_norm = (pred - pred_min) / (pred_max - pred_min + 1e-8)
        pred_norm = pred_norm * (real_max - real_min) + real_min

        # 存储数据
        real_list.append(real.flatten())
        pred_list.append(pred_norm.flatten())

        # 常规指标
        station_metrics['MAE'].append(mean_absolute_error(real, pred_norm))
        station_metrics['RMSE'].append(np.sqrt(mean_squared_error(real, pred_norm)))
        station_metrics['MAPE'].append(mean_absolute_percentage_error(real, pred_norm))
        station_metrics['JSD'].append(calculate_jsd(real.reshape(1, -1), pred_norm.reshape(1, -1)))

    # ===== MMD 计算 =====
    if len(real_list) > 1 and len(pred_list) > 1:
        real_array = np.vstack(real_list)
        pred_array = np.vstack(pred_list)
        mmd_score = calculate_mmd_stable(real_array, pred_array, sigma=50.0)
    else:
        mmd_score = 0.0

    result_row = {
        'station': station,
        'MAE': np.mean(station_metrics['MAE']),
        'RMSE': np.mean(station_metrics['RMSE']),
        'MAPE': np.mean(station_metrics['MAPE']),
        'MMD': mmd_score,
        'JSD': np.mean(station_metrics['JSD']),
    }
    results.append(result_row)

# ===================== 保存结果 =======================
results_df = pd.DataFrame(results)
results_df.to_csv('new_1W_成都_北京station评估.csv', index=False)
print("评估完成，结果保存至 new_1W_成都_北京station评估.csv")
