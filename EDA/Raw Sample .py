import os
import scipy.io

# 指定路径
data_path = r'G:\My Drive\Dataset\WildPPG'
mat_path = 'WildPPG_Part_fex.mat'
file_path = os.path.join(data_path, mat_path)


def load_wildppg_participant(path):
    """
    Loads the data of a WildPPG participant and cleans it to receive nested dictionaries.
    """
    # 加载 .mat 文件
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


# 调用函数加载数据
participant_data = load_wildppg_participant(file_path)

# 打印加载的基本信息
print(f"Participant ID: {participant_data['id']}")
print(f"Participant Notes: {participant_data['notes']}")

# 示例访问 sternum 部位的 PPG 数据
if participant_data['sternum']:
    ppg_g_data = participant_data['sternum']['acc_x']['v']
    print(f"PPG (Green) 数据点数量: {len(ppg_g_data)}")
    print(f"PPG (Green) 数据示例: {ppg_g_data[:10]}")  # 打印前 10 个数据点
else:
    print("Sternum 数据不可用。")
