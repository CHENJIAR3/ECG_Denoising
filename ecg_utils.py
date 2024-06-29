# 该文件夹放置一些 ECG信号预处理相关的代码

import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import resample
import scipy.io
from tqdm import tqdm
import biosppy.signals.ecg


# 文件路径
cpsc2020_data_path = "./Data_set/cpsc2020/TrainingSet/data/"
cpsc2020_ref_path = "./Data_set/cpsc2020/TrainingSet/ref/"


# 创建存放重采样结果的目录
cpsc2020_500hz_data_path = "./Data_set/cpsc2020_500hz/data/"
cpsc2020_500hz_ref_path = "./Data_set/cpsc2020_500hz/ref/"

"""
    将cps2020的数据集和R波标签由采样率400Hz、重新采样为500Hz
"""
def CPSC2020_Resample_to_500():
    # 新的采样率
    new_sampling_rate = 500

    # os.makedirs(result_folder, exist_ok=True)

    # 获取所有的心电数据.mat文件
    data_files = [file for file in os.listdir(cpsc2020_data_path) if file.endswith('.mat')]

    # 循环依次处理每个文件
    for data_file_name in tqdm(data_files[0:], desc="Processing Files", unit="step"):
        # 读取心电信号
        data_path = os.path.join(cpsc2020_data_path, data_file_name)
        ecg_mat_data = loadmat(data_path)
        ecg_signal = ecg_mat_data['ecg'].flatten()

        # 读取对应的r波标签
        ref_path = os.path.join(cpsc2020_ref_path, "RP" + str(int(data_file_name[1:-4])) + ".npy")
        r_peaks = np.load(ref_path)

        # 重采样至500Hz
        num_samples = len(ecg_signal)
        ecg_resampled = resample(ecg_signal, int(new_sampling_rate * (num_samples / 400)))
        r_peaks = np.round((r_peaks / 400) * 500).astype(int)

        # 保存重采样的数据和R波标签为.mat文件
        scipy.io.savemat(cpsc2020_500hz_data_path + data_file_name, {'ecg': ecg_resampled})
        scipy.io.savemat(cpsc2020_500hz_ref_path + data_file_name, {'r_peaks': r_peaks})
        


"""
    对CPSC2020 重采样后500Hz的信号进行R波位置的可视化呈现，并将呈现结果以图片形式进行存储
"""
cpsc2020_500hz_fig_path = "./Data_set/cpsc2020_500hz/fig/"

def CPSC2020_R_Visible():
       # 获取所有的心电数据.mat文件
    data_files = [file for file in os.listdir(cpsc2020_500hz_data_path) if file.endswith('.mat')]

    # 循环依次处理每个文件
    for data_file_name in tqdm(data_files[0:], desc="Processing Files", unit="step"):
        # 读取ecg数据和r波标签
        ecg_data = loadmat(cpsc2020_500hz_data_path+data_file_name)['ecg'].flatten() 
        r_peaks = loadmat(cpsc2020_500hz_ref_path+data_file_name)['r_peaks'].flatten() 

        # 创建存储图片的子文件夹
        fig_subfolder = os.path.join(cpsc2020_500hz_fig_path, data_file_name[:-4])
        os.makedirs(fig_subfolder, exist_ok=True)

        # 计算每个片段的时间范围
        segment_duration = 10  # 每段10秒
        num_segments = len(ecg_data) // (500 * segment_duration)  # 一段ecg片段按10s分割，则共有的ECG片段数量

        time_ranges = [(i * segment_duration, (i + 1) * segment_duration) for i in range(num_segments)]


        # 绘制图形并保存
        for i, (start_time, end_time) in enumerate(time_ranges):
            plt.figure(figsize=(12, 6))
            segment_indices = np.arange(start_time * 500, end_time * 500)

            plt.plot(segment_indices / 500, ecg_data[segment_indices])

            plot_r_peaks_range = (r_peaks >= segment_indices[0]) & (r_peaks < segment_indices[-1])
            plt.scatter((r_peaks[plot_r_peaks_range]) / 500,
                        
                        ecg_data[r_peaks[plot_r_peaks_range]],

                        color='red', marker='o', label='R peaks')
            
            plt.title(f'ECG Signal - {os.path.splitext(data_file_name)[0]} - Segment {i + 1}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)

            # 保存图像
            fig_name = f"{i + 1}.jpg"
            fig_path = os.path.join(fig_subfolder, fig_name)
            plt.savefig(fig_path, dpi=100)
            plt.close()





"""
    数据库: mit-bih-arrhythmia-database-1.0.0
    https://www.physionet.org/content/mitdb/1.0.0/
    48个半小时 双导联 MLII和V1导联  动态心电图记录
    数据采样率 360Hz, 11位数据分辨率、
"""
mitbih_arryth_ecg_names = ['100', '101', '102', '103', '104', '105', '106', '107', '108',
 '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
 '122', '123', '124', '200', '201','202', '203', '205', '207', '208', '209',
 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223',
 '228', '230', '231', '232', '233', '234']

mit_bih_ar_data_path = "./Data_set/mit-bih-arrhythmia-database-1.0.0/"
# 创建数据和标签存储文件夹
mit_bih_ar_500hz_data_path = './Data_set/mit_bih_ar_500hz/data/'
mit_bih_ar_500hz_ref_path = './Data_set/mit_bih_ar_500hz/ref/'

"""
    将 MIT_BIH_AR 数据集和R波标签由采样率360Hz、重采样到500Hz并保存为.mat文件
"""
def MIT_BIH_AR_Resample_to_500():
    # 新的采样率
    new_sampling_rate = 500
    for ecg_name in tqdm(mitbih_arryth_ecg_names[0:], desc="Resample Files", unit="step"):
        # 读取原始数据和标记信息
        ecg_data = wfdb.rdrecord(os.path.join(mit_bih_ar_data_path, ecg_name), physical=False).d_signal
        ecg_data = np.array(ecg_data).T 
        ecg_ann = wfdb.rdann(os.path.join(mit_bih_ar_data_path, ecg_name), extension="atr").__dict__

        r_peaks = ecg_ann['sample']  # R波位置 
        r_type = ecg_ann['symbol']   # R波类型
        original_sampling_rate = ecg_ann['fs'] # 获取原始采样率
   
        or_sig_len = ecg_data.shape[1]     # 获取通道数据的长度

        # 重采样至新的采样率
        resampled_mlii_ecg = resample(ecg_data[0], int(new_sampling_rate*(or_sig_len/original_sampling_rate)))
        resampled_v1_ecg = resample(ecg_data[1], int(new_sampling_rate*(or_sig_len/original_sampling_rate)))

        # 重采样 R 波标记
        resampled_r_peaks = [int((peak*new_sampling_rate)/original_sampling_rate) for peak in r_peaks]
        
        # 保存重采样后的心电数据为.mat文件
        data_save_path = os.path.join(mit_bih_ar_500hz_data_path, f'{ecg_name}.mat')
        scipy.io.savemat(data_save_path, {'mlii_ecg':resampled_mlii_ecg,'v1_ecg':resampled_v1_ecg})
        # 保存重采样后的 R 波标记为.mat文件
        ref_save_path = os.path.join(mit_bih_ar_500hz_ref_path, f'{ecg_name}.mat')
        scipy.io.savemat(ref_save_path, {'r_peaks': resampled_r_peaks})



"""
    对 MIT_BIH_AR 数据集重采样后500Hz的信号进行R波位置的可视化呈现，并将可视化结果以图片形式进行存储
    R 波位置可视化包含：
                    1、真实的R波位置
                    2、Hamilton R波定位算法定位的R波位置
"""
mit_bih_ar_500hz_fig_path = './Data_set/mit_bih_ar_500hz/fig/'
def MIT_BIH_AR_R_Visible():
    # 循环依次处理每个文件
    # 循环依次处理每个文件
    for ecg_name in tqdm(mitbih_arryth_ecg_names, desc="Processing Files", unit="step"):
        # 读取重采样后的ecg数据和r波标签
        ecg_data_path = os.path.join(mit_bih_ar_500hz_data_path, f'{ecg_name}.mat')
        r_peaks_path = os.path.join(mit_bih_ar_500hz_ref_path, f'{ecg_name}.mat')

        ecg_data = scipy.io.loadmat(ecg_data_path)
        r_peaks = scipy.io.loadmat(r_peaks_path)['r_peaks'].flatten()

        # 创建存储图片的子文件夹
        fig_subfolder = os.path.join(mit_bih_ar_500hz_fig_path, ecg_name)
        os.makedirs(fig_subfolder, exist_ok=True)
        
        # 分别获得原始记录中，的两个导联的数据
        mlii_ecg = ecg_data['mlii_ecg'].flatten()
        v1_ecg = ecg_data['v1_ecg'].flatten()

        # 对ECG信号应用 Hamilton R波检测算法
        mlii_ecg_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(mlii_ecg, sampling_rate=500)[0]
        v1_ecg_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(v1_ecg, sampling_rate=500)[0]


        # 计算每个片段的时间范围
        segment_duration = 10  # 每段10秒
        num_segments = len(mlii_ecg) // (500 * segment_duration)  # 一段ecg片段按10s分割，则共有的ECG片段数量
        time_ranges = [(i * segment_duration, (i + 1) * segment_duration) for i in range(num_segments)]

        # 绘制图形并保存
        for seg_i, (start_time, end_time) in enumerate(time_ranges):
            plt.figure(figsize=(12, 6))

            # Subplot for lead mlii
            plt.subplot(2, 1, 1)
            segment_indices = np.arange(start_time * 500, end_time * 500)

            plt.plot(segment_indices / 500, mlii_ecg[segment_indices])
            true_plot_r_peaks_range = (r_peaks >= segment_indices[0]) & (r_peaks < segment_indices[-1])

            ham_mlii_r_range = (mlii_ecg_Hamilton_rpeaks >= segment_indices[0]) & (mlii_ecg_Hamilton_rpeaks < segment_indices[-1])
            ham_v1_r_range = (v1_ecg_Hamilton_rpeaks >= segment_indices[0]) & (v1_ecg_Hamilton_rpeaks < segment_indices[-1])

            plt.scatter((r_peaks[true_plot_r_peaks_range]) / 500,
                        mlii_ecg[r_peaks[true_plot_r_peaks_range]],
                        color='red', marker='o', label='R peaks')
            
            plt.scatter(mlii_ecg_Hamilton_rpeaks[ham_mlii_r_range] / 500, 
                        mlii_ecg[mlii_ecg_Hamilton_rpeaks[ham_mlii_r_range]], 
                        color='blue', label=' Hamilton R Peaks', marker='H')

            plt.title(f'ECG Signal - {ecg_name} - Seg {seg_i + 1} - Lead mlii')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)


            # Subplot for lead v1
            plt.subplot(2, 1, 2)
            plt.plot(segment_indices / 500, v1_ecg[segment_indices])
            plt.scatter((r_peaks[true_plot_r_peaks_range]) / 500,
                        v1_ecg[r_peaks[true_plot_r_peaks_range]],
                        color='red', marker='o', label='R peaks')
            
            plt.scatter(v1_ecg_Hamilton_rpeaks[ham_v1_r_range]/500, 
                        v1_ecg[v1_ecg_Hamilton_rpeaks[ham_v1_r_range]], 
                        color='black', label=' Hamilton R Peaks', marker='H')
            
            plt.title(f'ECG Signal - {ecg_name} - Seg {seg_i + 1} - Lead v1')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)

            # Adjust layout
            plt.tight_layout()
            # 保存图像
            fig_name = f"{ecg_name}_Seg_{seg_i + 1}.jpg"
            fig_path = os.path.join(fig_subfolder, fig_name)
            plt.savefig(fig_path, dpi=100)
            plt.close()




if __name__ == "__main__":
    # CPSC2020_Resample_to_500()
    # CPSC2020_R_Visible()
    # MIT_BIH_AR_Resample_to_500()
    MIT_BIH_AR_R_Visible()




