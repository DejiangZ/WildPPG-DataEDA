import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# 加载 WildPPG 数据的函数
def load_wildppg_participant(path):
    """
    Loads the data of a WildPPG participant and cleans it to receive nested dictionaries.
    """
    loaded_data = scipy.io.loadmat(path)

    # 清理 id 字段
    loaded_data['id'] = loaded_data['id'][0]

    # 清理 notes 字段
    if len(loaded_data['notes']) == 0:
        loaded_data['notes'] = ""
    else:
        loaded_data['notes'] = loaded_data['notes'][0]

    # 清理每个身体部位的数据
    for bodyloc in ['sternum', 'head', 'wrist', 'ankle']:
        bodyloc_data = dict()  # 用于存储清理后的数据
        if bodyloc in loaded_data:
            sensors = loaded_data[bodyloc][0].dtype.names  # 获取传感器名称
            for sensor_name, sensor_data in zip(sensors, loaded_data[bodyloc][0][0]):
                bodyloc_data[sensor_name] = dict()
                field_names = sensor_data[0][0].dtype.names  # 获取传感器字段名称
                for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                    bodyloc_data[sensor_name][sensor_field] = field_data[0]
                    if sensor_field == 'fs':  # 如果是采样率字段，提取标量值
                        bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][sensor_field][0]
            loaded_data[bodyloc] = bodyloc_data  # 替换为清理后的数据
        else:
            loaded_data[bodyloc] = None  # 如果该身体部位没有数据，则设置为 None

    return loaded_data

# 指定路径
data_path = r'G:\My Drive\Dataset\WildPPG'
mat_path = 'WildPPG_Part_an0.mat'
file_path = os.path.join(data_path, mat_path)

# 加载数据
participant_data = load_wildppg_participant(file_path)

# 计算加速度幅值
def calculate_acceleration_magnitude(acc_x, acc_y, acc_z):
    """
    计算加速度幅值。
    """
    return np.sqrt(np.square(acc_x) + np.square(acc_y) + np.square(acc_z))

if participant_data['wrist']:
    acc_x = participant_data['wrist']['acc_x']['v']
    acc_y = participant_data['wrist']['acc_y']['v']
    acc_z = participant_data['wrist']['acc_z']['v']

    # 确保三个数组长度一致
    min_length = min(len(acc_x), len(acc_y), len(acc_z))
    acc_x, acc_y, acc_z = acc_x[:min_length], acc_y[:min_length], acc_z[:min_length]

    # 计算加速度幅值
    acc_magnitude = calculate_acceleration_magnitude(acc_x, acc_y, acc_z)

    # 可视化加速度幅值
    plt.figure(figsize=(12, 6))
    plt.plot(acc_magnitude, label="Acceleration Magnitude")
    plt.title("Acceleration Magnitude Over Time (wrist)")
    plt.xlabel("Time Index")
    plt.ylabel("Acceleration Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Sternum data is not available.")
