import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from datetime import datetime
import torch
import statsmodels.api as sm
import matplotlib.pyplot as plt

class ChannelAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(num_channels, num_channels)
        self.fc2 = nn.Linear(num_channels, num_channels)
        
    def forward(self, x):
        # x shape: (batch_size, num_channels, timesteps, dimensions)
        
        # Global average pooling
        avg_pool = torch.mean(x, dim=[2, 3])  # shape: (batch_size, num_channels)
        
        # Channel attention scores
        attention = F.relu(self.fc1(avg_pool))  # shape: (batch_size, num_channels)
        attention = torch.sigmoid(self.fc2(attention))  # shape: (batch_size, num_channels)
        
        return attention

def stl_decomposition(tensor, period):
    """
    对每个变量进行STL分解
    
    参数：
        tensor (torch.Tensor): 输入时间序列张量，形状为 [batch_size, timesteps, features]
        period (int): 季节周期
    
    返回：
        trend_tensor (torch.Tensor): 趋势项张量
        seasonal_tensor (torch.Tensor): 季节项张量
        residual_tensor (torch.Tensor): 残差项张量
    """
    tensor_np = tensor.cpu().numpy()
    batch_size, timesteps, features = tensor_np.shape

    trend_list, seasonal_list, residual_list = [], [], []

    for i in range(features):
        trend_feature, seasonal_feature, residual_feature = [], [], []
        for j in range(batch_size):
            series = tensor_np[j, :, i]
            stl = sm.tsa.STL(series, period=period, robust=True).fit()
            trend_feature.append(stl.trend)
            seasonal_feature.append(stl.seasonal)
            residual_feature.append(stl.resid)
        trend_list.append(trend_feature)
        seasonal_list.append(seasonal_feature)
        residual_list.append(residual_feature)

    trend_tensor = torch.tensor(np.array(trend_list)).permute(1, 2, 0).to(tensor.device)
    seasonal_tensor = torch.tensor(np.array(seasonal_list)).permute(1, 2, 0).to(tensor.device)
    residual_tensor = torch.tensor(np.array(residual_list)).permute(1, 2, 0).to(tensor.device)

    return trend_tensor, seasonal_tensor, residual_tensor

def calculate_strength(trend_tensor, seasonal_tensor, residual_tensor):
    var_r = torch.var(residual_tensor, dim=1, unbiased=False)
    var_t_r = torch.var(trend_tensor + residual_tensor, dim=1, unbiased=False)
    var_s_r = torch.var(seasonal_tensor + residual_tensor, dim=1, unbiased=False)

    ft = torch.max(torch.tensor(0.0, device=residual_tensor.device), 1 - var_r / var_t_r)
    fs = torch.max(torch.tensor(0.0, device=residual_tensor.device), 1 - var_r / var_s_r)
    
    return ft.mean().item(), fs.mean().item()

def calculate_overall_strength(tensor, period):
    trend_tensor, seasonal_tensor, residual_tensor = stl_decomposition(tensor, period)
    trend_strength, seasonal_strength = calculate_strength(trend_tensor, seasonal_tensor, residual_tensor)
    return trend_strength, seasonal_strength

def calculate_statistics(x):
    """
    计算输入时间序列数据的均值和方差。

    参数：
    x (torch.Tensor): 输入的时间序列数据，形状为 (batch_size, time_steps, features)

    返回：
    tuple: 包含均值和方差的元组，每个形状为 (features,)
    """
    # 将张量从 GPU 移动到 CPU 并转换为 NumPy 数组
    x_np = x.cpu().detach().numpy()
    
    means = np.mean(x_np, axis=(0, 1))  # 计算每个特征的均值
    stds = np.std(x_np, axis=(0, 1))    # 计算每个特征的标准差
    return means, stds

def generate_gaussian_noise(means, stds, batch_size, time_steps):
    """
    根据均值和方差生成模拟时间序列数据。

    参数：
    means (numpy.ndarray): 每个特征的均值，形状为 (features,)
    stds (numpy.ndarray): 每个特征的标准差，形状为 (features,)
    batch_size (int): 生成数据的批次大小
    time_steps (int): 生成数据的时间步数

    返回：
    numpy.ndarray: 生成的模拟时间序列数据，形状为 (batch_size, time_steps, features)
    """
    features = means.shape[0]
    synthetic_data = np.random.normal(loc=means, scale=stds, size=(batch_size, time_steps, features))
    synthetic_data_tensor = torch.tensor(synthetic_data, dtype=torch.float32)
    return synthetic_data_tensor

def plot_time_series_tensor(tensor, display_plot=True):
    # Check if the input is a PyTorch tensor and convert it to a NumPy array
    

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Plotting the tensor
    plt.figure()
    #for i in range(tensor.shape[1]):
    plt.plot(tensor[:, 0])
    
    #plt.title('Time Series Tensor Plot')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # Save the plot without a legend
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'autotcl\time_series_tensor_plot_{timestamp}.png'
    plt.savefig(file_name, bbox_inches='tight')
    
    # Display the plot if required
    if display_plot:
        plt.show()
    
    plt.close()

    return file_name

def plot_time_series_batches(x, ax, function, batch_index=0):
    """
    绘制两个给定批次的时间序列图像。
    
    参数：
    x (torch.Tensor): 第一个时间序列数据，形状为 [batch_size, time_steps, features]
    ax (torch.Tensor): 第二个时间序列数据，形状为 [batch_size, time_steps, features]
    batch_index (int): 要绘制的批次索引，默认为 0
    """
    # 检查输入的形状
    if len(x.shape) != 3 or len(ax.shape) != 3:
        raise ValueError("输入张量 x 和 ax 的形状必须为 [batch_size, time_steps, features]")
    
    if x.shape != ax.shape:
        print("x.shape", x.shape)
        print("ax.shape", ax.shape)
        raise ValueError("输入张量 x 和 ax 必须具有相同的形状")
    
    batch_size, time_steps, features = x.shape

    # 检查批次索引的有效性
    if batch_index < 0 or batch_index >= batch_size:
        raise ValueError(f"批次索引必须在 0 和 {batch_size-1} 之间")

    # 将张量移动到 CPU 并分离
    x_cpu = x.cpu().detach()
    ax_cpu = ax.cpu().detach()

    # 选择指定批次的数据
    batch_data_x = x_cpu[batch_index]
    batch_data_ax = ax_cpu[batch_index]

    # 创建图形
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # 绘制第一个张量的时间序列
    for feature in range(features):
        axs[0].plot(range(time_steps), batch_data_x[:, feature].numpy(), label=f'Feature {feature + 1}')
    axs[0].legend()
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Value')
    axs[0].set_title(f'{function}: Time Series for Batch {batch_index + 1} (x)')

    # 绘制第二个张量的时间序列
    for feature in range(features):
        axs[1].plot(range(time_steps), batch_data_ax[:, feature].numpy(), label=f'Feature {feature + 1}')
    axs[1].legend()
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Value')
    axs[1].set_title(f'Time Series for Batch {batch_index + 1} (ax)')

    # 显示图形
    plt.tight_layout()
    plt.show()


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        print("SEED ",seed)
        seed += 1
        np.random.seed(seed)
        print("SEED ",seed)
        seed += 1
        torch.manual_seed(seed)
        print("SEED ",seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
                print("SEED " , seed)

    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]


def plot2d(x, y, x2=None, y2=None, x3=None, y3=None, xlim=(-1, 1), ylim=(-1, 1), save_file=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    plt.plot(x, y)
    if x2 is not None and y2 is not None:
        plt.plot(x2, y2)
    if x3 is not None and y3 is not None:
        plt.plot(x3, y3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, "")
    else:
        plt.show()
    return

def plot1d(x, x2=None, x3=None, ylim=(-1, 1), save_file=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3))
    steps = np.arange(x.shape[0])
    plt.plot(steps, x)
    if x2 is not None:
        plt.plot(steps, x2)
    if x3 is not None:
        plt.plot(steps, x3)
    plt.xlim(0, x.shape[0])
    plt.ylim(ylim)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()
    return

class dict2class:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def plot_time_series(data, num_time_steps=50):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    绘制时间序列数据的前 num_time_steps 个时间步，并使用不同颜色区分不同变量。

    Args:
    - data (numpy.ndarray): 输入时间序列数据，形状为 (batch_size, sequence_length, num_features)。
    - num_time_steps (int): 要显示的时间步长数量。

    Returns:
    - None
    """

    # 提取前 num_time_steps 个时间步的数据
    if num_time_steps > data.shape[1]:
        num_time_steps = data.shape[1]

    data_subset = data[:, :num_time_steps, :]

    # 创建时间轴
    time_steps = np.arange(num_time_steps)

    # 绘制时间序列图
    plt.figure(figsize=(15, 6))

    for i in range(data_subset.shape[2]):
        plt.plot(time_steps, data_subset[0, :, i], label=f'Feature {i+1}')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'Time Series Data (First {num_time_steps} Time Steps)')
    plt.legend()
    plt.grid(True)
    plt.show()