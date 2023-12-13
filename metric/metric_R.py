# This file is used to calculate the change of R-wave calculation accuracy before and after signal denoising
import os
import math
import pandas as pd
import numpy as np
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt
import biosppy.signals.ecg
from ecgdetectors import Detectors
from sklearn.metrics import precision_score, recall_score, f1_score

"""
    cpsc2018  500Hz采样率 12导联心律失常分类任务
    Normal	AF	I-AVB	LBBB	RBBB	PAC	PVC	STD	STE
    1       2    3      4        5      6    7   8   9
    http://2018.icbeb.org/Challenge.html

"""




"""
    cpsc2019 R波识别任务  fs 500Hz  10s的ecg片段
    http://2019.icbeb.org/Challenge.html
"""

cpsc2019_data_path = "./Data_set/cpsc2019/train/data/"
cpsc2019_ref_path = "./Data_set/cpsc2019/train/ref/"
cpsc2019_R_figpath = "./results/cpsc2019_R_result/R_fig/"

cpsc2019_R_mat = "./results/cpsc2019_R_result/R_mat/"




'''
# CPSC2019 R波识别可视化
'''
def CPSC2019R_Visible():

    # 获取所有的.mat文件
    data_files = [file for file in os.listdir(cpsc2019_data_path) if file.endswith('.mat')]

    # 循环处理
    for data_file_name in data_files[0:20]:
        ref_file_name = 'R'+data_file_name[4:]
        # 读取数据
        ecg_data = scipy.io.loadmat(cpsc2019_data_path+data_file_name)['ecg']
        r_peak_data = scipy.io.loadmat(cpsc2019_ref_path+ref_file_name)['R_peak'].flatten()

        # 计算时间轴
        sampling_rate = 500  # 采样率为500Hz
        duration = 10  # 信号长度为10秒
        time = np.arange(0, duration, 1/sampling_rate)

        # 绘制ECG图
        plt.figure(figsize=(12, 6))
        plt.plot(time, ecg_data, label='ECG Signal')
        plt.scatter(r_peak_data/sampling_rate, ecg_data[r_peak_data], color='red', label='R Peaks', marker='x')
        plt.title('ECG Signal with R Peak Markers')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        # plt.show()
        plt.savefig(cpsc2019_R_figpath+data_file_name[:-4]+'.jpg',dpi=600)
        plt.close()



'''
# CPSC2019 R波识别、存储及可视化

pip install py-ecg-detectors 这个库里面包含了很多r波检测函数
https://pypi.org/project/py-ecg-detectors/

'''
def CPSC2019_R_Visible():

    # 获取所有的.mat文件
    data_files = [file for file in os.listdir(cpsc2019_data_path) if file.endswith('.mat')]

    # 循环处理
    for data_file_name in data_files[0:20]:
        ref_file_name = 'R'+data_file_name[4:]
        # 读取数据
        ecg_data = scipy.io.loadmat(cpsc2019_data_path+data_file_name)['ecg']
        r_peak_data = scipy.io.loadmat(cpsc2019_ref_path+ref_file_name)['R_peak'].flatten()

        # 信号长度和采样率
        signal_length = len(ecg_data)

        # 对ECG信号应用Hamilton R波检测算法
        Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(ecg_data.T[0], sampling_rate=500)[0]
     

        # 对ECG信号应用Pan-Tompkins R波检测算法 
        detectors = Detectors(500)  # 500是采样率
        Pan_rpeaks = detectors.pan_tompkins_detector(ecg_data.T[0])

        # 保存R波检测的结果
        scipy.io.savemat(cpsc2019_R_mat+ref_file_name, {'Pan_rpeaks':Pan_rpeaks,'Hamilton_rpeaks':Hamilton_rpeaks})


        # # 对ECG信号应用 engzee_detector R波检测算法 
        # rpeaks_engzee = detectors.engzee_detector(ecg_data.T[0])


        # 计算时间轴
        sampling_rate = 500  # 采样率为500Hz
        duration = 10  # 信号长度为10秒
        time = np.arange(0, duration, 1/sampling_rate)

        # 绘制ECG图
        plt.figure(figsize=(12, 6))
        plt.plot(time, ecg_data, label='ECG Signal')
        plt.scatter(r_peak_data/sampling_rate, ecg_data[r_peak_data], color='red', label='R Peaks', marker='x')

        plt.scatter(Hamilton_rpeaks/sampling_rate, ecg_data[Hamilton_rpeaks], color='blue', label=' Hamilton  Detected R Peaks', marker='H')

        plt.scatter([x/sampling_rate for x in Pan_rpeaks], ecg_data[Pan_rpeaks], color='black', label='Detected R Peaks (Pan-Tompkins)', marker='s')

        # plt.scatter([x/sampling_rate for x in rpeaks_engzee], ecg_data[rpeaks_engzee], color='black', label='engzee_detector)', marker='o')

        plt.grid(which='minor', alpha=0.2,color='r')
        plt.grid(which='major', alpha=0.5,color='r')  
        plt.title('ECG Signal with R Peak Markers')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        # plt.show()
        plt.savefig(cpsc2019_R_figpath+data_file_name[:-4]+'.jpg',dpi=600)
        plt.close()








"""
使用 detectors 库中的各种R波检测算法、检测R波并画图标记R波的位置

"""
# def CPSC2019_R_locate():

#     # 获取所有的.mat文件
#     data_files = [file for file in os.listdir(cpsc2019_data_path) if file.endswith('.mat')]

#         # 循环处理
#     for data_file_name in data_files[0:20]:
#         ref_file_name = 'R'+data_file_name[4:]
#         # 读取数据
#         ecg_data = scipy.io.loadmat(cpsc2019_data_path+data_file_name)['ecg']
#         r_peak_data = scipy.io.loadmat(cpsc2019_ref_path+ref_file_name)['R_peak'].flatten()

#         # 定义采样率
#         sampling_rate = 500

#         # 创建Detectors对象
#         detectors = Detectors(sampling_rate)

#         # 定义所有的检测器
#         detector_functions = [
#             detectors.hamilton_detector,
#             detectors.christov_detector,
#             detectors.engzee_detector,
#             detectors.pan_tompkins_detector,
#             detectors.swt_detector,
#             detectors.two_average_detector,
#             detectors.wqrs_detector
#         ]

#         # 定义对应的算法名称
#         detector_names = [
#             'Hamilton',
#             'Christov',
#             'Engzee',
#             'Pan-Tompkins',
#             'SWT',
#             'Two Average',
#             'WQRS'
#         ]

#         # 绘制ECG图并标记R波位置
#         plt.figure(figsize=(18, 12))
#         plt.plot(ecg_data, label='ECG Signal')
#         plt.scatter(r_peak_data, ecg_data[r_peak_data], color='red', label='True R Peaks', marker='x')

#         # 遍历每个检测器
#         for detector_func, detector_name in zip(detector_functions, detector_names):
#             # 使用检测器进行R波检测
#             rpeaks = detector_func(ecg_data.T[0])

#             # 绘制检测的R波位置
#             plt.scatter([x/sampling_rate for x in rpeaks], ecg_data[rpeaks], label=f'Detected R Peaks ({detector_name})', marker='o')

#         plt.title('ECG Signal with R Peaks Detection using Various Algorithms')
#         plt.xlabel('Sample Index')
#         plt.ylabel('Amplitude')
#         plt.legend()
#         plt.savefig(cpsc2019_R_figpath+data_file_name[:-4]+'.jpg',dpi=600)
#         # plt.show()
#         plt.close()







"""
    该函数同时使用多种R波定位算法，对原始信号和抗噪信号进行R波定位，并记录R波定位结果
"""

Denoised_cpsc2019_mat = "./results/Denoised_cpsc2019/mat/"  # 抗噪后的数据文件路径
cpsc2019_De_R_mat = "./results/cpsc2019_R_result/De_R_mat/"
cpsc2019_De_R_fig = "./results/cpsc2019_R_result/De_R_fig/"


def CPSC2019_or_de_R_Locate():
    # 获取所有的.mat文件
    sampling_rate = 500  # 采样率为500Hz
    data_files = [file for file in os.listdir(Denoised_cpsc2019_mat) if file.endswith('.mat')]

    # 循环依次处理每个文件
    print("total files",len(data_files))
    for data_file_name in tqdm(data_files[0:], desc="Processing Files", unit="step"):
        ref_file_name = 'R'+data_file_name[4:]
        # 读取数据
        ecg_mat_data = scipy.io.loadmat(Denoised_cpsc2019_mat+data_file_name)
        or_ecg_data = ecg_mat_data['ecg_orl']  # 原始数据
        de_ecg_data = ecg_mat_data['ecg_de']   # 去噪后的数据
        true_r_peak_data = scipy.io.loadmat(cpsc2019_ref_path+ref_file_name)['R_peak'].flatten()   # 真实R波标签数据


        # 对ECG信号应用 Hamilton R波检测算法
        or_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(or_ecg_data.T[0], sampling_rate=500)[0]
        de_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(de_ecg_data.T[0], sampling_rate=500)[0]


        # 对ECG信号应用Pan-Tompkins R波检测算法 
        detectors = Detectors(500)  # 500是采样率
        or_Pan_rpeaks = detectors.pan_tompkins_detector(or_ecg_data.T[0])
        de_Pan_rpeaks = detectors.pan_tompkins_detector(de_ecg_data.T[0])
        
        # 保存R波检测的结果
        scipy.io.savemat(cpsc2019_De_R_mat+ref_file_name, {'true_r_peak_data':true_r_peak_data, 'or_Hamilton_rpeaks':or_Hamilton_rpeaks,'de_Hamilton_rpeaks':de_Hamilton_rpeaks \
                        ,'or_Pan_rpeaks':or_Pan_rpeaks, 'de_Pan_rpeaks':de_Pan_rpeaks})



        # 图片绘制、及R波定位结果可视化
        a=plt.figure()
        a.set_size_inches(12, 10)
        ax=plt.subplot(211)
        major_ticksx = np.arange(0, 10*sampling_rate, 1*sampling_rate)
        minor_ticksx = np.arange(0, 10*sampling_rate, 0.25*sampling_rate)

        max = np.max(or_ecg_data) 
        min = np.min(or_ecg_data) 
        delet = max-min
        max = math.ceil(max + delet*0.25)
        min = math.floor(min - delet*0.25)

        major_ticksy = np.arange(min, max,0.3*delet)
        minor_ticksy = np.arange(min, max, 0.075*delet)  
        ax.set_xticks(major_ticksx)
        ax.set_xticks(minor_ticksx, minor=True)          
        ax.set_yticks(major_ticksy)
        ax.set_yticks(minor_ticksy, minor=True)
        plt.plot(or_ecg_data,linewidth=0.7,color='k')  # 绘制原始数据
        # 绘制原始数据R波定位结果
        plt.scatter(true_r_peak_data, or_ecg_data[true_r_peak_data], color='red', label='R Peaks', marker='x')
        plt.scatter(or_Hamilton_rpeaks, or_ecg_data[or_Hamilton_rpeaks], color='blue', label=' Hamilton  Detected R Peaks', marker='H')
        plt.scatter(or_Pan_rpeaks, or_ecg_data[or_Pan_rpeaks], color='black', label='Detected R Peaks (Pan-Tompkins)', marker='s')
        plt.legend()


        ax.grid(which='minor', alpha=0.2,color='r')
        ax.grid(which='major', alpha=0.5,color='r')  
        plt.title("Original ECG", fontsize=15)
        # plt.axis([0, 10,-1.5, 1.5])
        plt.xlabel('Time ', fontsize=13)
        plt.ylabel('Amplitude', fontsize=13)
        ax2=plt.subplot(212, sharex = ax)
        major_ticksx = np.arange(0, 10*sampling_rate,1*sampling_rate )
        minor_ticksx = np.arange(0, 10*sampling_rate, 0.25*sampling_rate)
        max = np.max(de_ecg_data) 
        min = np.min(de_ecg_data) 
        delet = max-min
        max = math.ceil(max + delet*0.25)
        min = math.floor(min - delet*0.25)
        major_ticksy = np.arange(min, max,0.3*delet)
        minor_ticksy = np.arange(min, max, 0.075*delet)  
        ax2.set_xticks(major_ticksx)
        ax2.set_xticks(minor_ticksx, minor=True)         
        ax2.set_yticks(major_ticksy)
        ax2.set_yticks(minor_ticksy, minor=True)
        plt.plot(de_ecg_data, linewidth=0.7,color='k')
        # 绘制抗噪后数据R波定位结果
        plt.scatter(true_r_peak_data, de_ecg_data[true_r_peak_data], color='red', label='R Peaks', marker='x')
        plt.scatter(de_Hamilton_rpeaks, de_ecg_data[de_Hamilton_rpeaks], color='blue', label=' Hamilton  Detected R Peaks', marker='H')
        plt.scatter(de_Pan_rpeaks, de_ecg_data[de_Pan_rpeaks], color='black', label='Detected R Peaks (Pan-Tompkins)', marker='s')


        ax2.grid(which='minor', alpha=0.2,color='r')
        ax2.grid(which='major', alpha=0.5,color='r')  
        plt.title("Denoised ECG", fontsize=15)
        # plt.axis([0, 10,-1.5, 1.5])
        plt.xlabel('Time', fontsize=13)
        plt.ylabel('Amplitude', fontsize=13)
        plt.savefig(cpsc2019_De_R_fig+data_file_name[:-4]+'.jpg',dpi=100) # 一般的屏幕显示dip100将足够了、科学出版物用dip600左右比较好  
        plt.close()




def calculate_R_metrics(true_peaks, detected_peaks, sampling_rate=500, tolerance=0.075):
    """
    计算 R 波定位的性能指标。

    参数:
    - true_peaks: 真实的 R 波位置列表
    - detected_peaks: 算法检测到的 R 波位置列表
    - sampling_rate: 信号的采样率，默认为500Hz
    - tolerance: 允许的时间容忍度，以秒为单位，默认为75ms
    """
    tolerance_samples = int(tolerance * sampling_rate)
    
    TP = sum(any(abs(true_peak - detected_peaks) <= tolerance_samples) for true_peak in true_peaks)
    FN = len(true_peaks) - TP
    FP = len(detected_peaks) - TP

    return TP, FN, FP

def print_metrics_table(title, metrics):
    columns = ["title", "Total TP", "Total FN", "Total FP", "Precision", "Recall", "F1 Score"]
    # 将title 作为新列添加到 metrics 列表
    metrics_with_title = [title] + list(metrics) 
    df = pd.DataFrame([metrics_with_title], columns=columns)
    
    print(title)
    print(df)
    print("\n")
    return df


def calculate_total_metrics(TP_list, FN_list, FP_list):
    total_TP = sum(TP_list)
    total_FN = sum(FN_list)
    total_FP = sum(FP_list)

    # # 计算性能指标
    # Precision = TP / (TP + FP) if TP + FP > 0 else 0 # 查准率：查准率等于预测正确的正样本数量/所有预测为正样本数量
    # Recall = TP / (TP + FN) if TP + FN > 0 else 0  # 查全率 ：预测正确的正样本数量/所有正样本的总（Recall越大说明漏检的越少，Recall越小说明漏检的越多。）
    # F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall > 0 else 0
    Precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    Recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall > 0 else 0

    return total_TP, total_FN, total_FP, Precision, Recall, F1




"""
    该函数计算去噪前后 R波定位的性能
"""
def CPSC2019_R_Metric_Calcu():
    # 原始的信号使用 Hamilton 定位R波的情况
    or_Ham_R_TP = []
    or_Ham_R_FN = []
    or_Ham_R_FP = []
    # 去噪后的信号使用 Hamilton 定位R波的情况
    de_Ham_R_TP = []
    de_Ham_R_FN = []
    de_Ham_R_FP = []

    # 原始的信号使用 Hamilton 定位R波的情况
    or_PanT_R_TP = []
    or_PanT_R_FN = []
    or_PanT_R_FP = []
    # 去噪后的信号使用 Hamilton 定位R波的情况
    de_PanT_R_TP = []
    de_PanT_R_FN = []
    de_PanT_R_FP = []

    # 获取所有的.mat文件
    data_files = [file for file in os.listdir(cpsc2019_De_R_mat) if file.endswith('.mat')]

    # 循环依次处理每个文件
    print("total files",len(data_files))
    for data_file_name in tqdm(data_files[0:], desc="Processing Files", unit="step"):
        # 读取数据
        ecg_R_mat_data = scipy.io.loadmat(cpsc2019_De_R_mat+data_file_name)

        true_r_peak_data = ecg_R_mat_data['true_r_peak_data'].flatten()  
        or_Hamilton_rpeaks = ecg_R_mat_data['or_Hamilton_rpeaks'].flatten()   
        de_Hamilton_rpeaks = ecg_R_mat_data['de_Hamilton_rpeaks'].flatten()    
        or_Pan_rpeaks = ecg_R_mat_data['or_Pan_rpeaks'].flatten()   
        de_Pan_rpeaks = ecg_R_mat_data['de_Pan_rpeaks'].flatten()   
    
        ## 计算去噪前后，R波定位的性能指标
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,or_Hamilton_rpeaks)
        or_Ham_R_TP.append(TP)
        or_Ham_R_FN.append(FN)
        or_Ham_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,de_Hamilton_rpeaks)
        de_Ham_R_TP.append(TP)
        de_Ham_R_FN.append(FN)
        de_Ham_R_FP.append(FP)

        ## 计算去噪前后，R波定位的性能指标
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,or_Pan_rpeaks)
        or_PanT_R_TP.append(TP)
        or_PanT_R_FN.append(FN)
        or_PanT_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,de_Pan_rpeaks)
        de_PanT_R_TP.append(TP)
        de_PanT_R_FN.append(FN)
        de_PanT_R_FP.append(FP)



    # 计算总的性能指标
    or_Hamilton_metrics = calculate_total_metrics(or_Ham_R_TP, or_Ham_R_FN, or_Ham_R_FP)
    de_Hamilton_metrics = calculate_total_metrics(de_Ham_R_TP, de_Ham_R_FN, de_Ham_R_FP)
    or_PanT_metrics = calculate_total_metrics(or_PanT_R_TP, or_PanT_R_FN, or_PanT_R_FP)
    de_PanT_metrics = calculate_total_metrics(de_PanT_R_TP, de_PanT_R_FN, de_PanT_R_FP)

    # 打印结果
    df1 = print_metrics_table("Original Ham Metrics", or_Hamilton_metrics)
    df2 = print_metrics_table("Denoised Ham Metrics", de_Hamilton_metrics)
    df3 = print_metrics_table("Original PanT Metrics", or_PanT_metrics)
    df4 = print_metrics_table("Denoised PanT Metrics", de_PanT_metrics)

    # 保存识别到的指标为表格
    with pd.ExcelWriter('./metric/CPSC2019_R_Metric_Calcu.xlsx', engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 2) 
        df3.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 4) 
        df4.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 6) 







"""
    该函数同时使用多种R波定位算法，对CPSC2020数据库原始信号和抗噪信号进行R波定位，并记录R波定位结果
"""
Denoised_cpsc2020_mat = "./results/Denoised_cpsc2020/mat/"  # 抗噪后的数据文件路径
cpsc2020_500hz_ref_path = "./Data_set/cpsc2020_500hz/ref/"

cpsc2020_De_R_mat = "./results/cpsc2020_R_result/De_R_mat/"


def CPSC2020_or_de_R_Locate():
    # 获取所有的.mat文件
    data_files = [file for file in os.listdir(Denoised_cpsc2020_mat) if file.endswith('.mat')]

    # 循环依次处理每个文件
    print("total files",len(data_files))
    for data_file_name in tqdm(data_files[0:], desc="Processing CPSC2020 R_Locate", unit="step"):
        # 读取数据
        ecg_mat_data = scipy.io.loadmat(Denoised_cpsc2020_mat+data_file_name)
        or_ecg_data = ecg_mat_data['ecg_orl'].flatten()   # 原始数据
        de_ecg_data = ecg_mat_data['ecg_de'].flatten()    # 去噪后的数据

        true_r_peak_data = scipy.io.loadmat(cpsc2020_500hz_ref_path+data_file_name)['r_peaks'].flatten()   # 真实R波标签数据

        # 对ECG信号应用 Hamilton R波检测算法
        or_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(or_ecg_data, sampling_rate=500)[0]
        de_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(de_ecg_data, sampling_rate=500)[0]


        # 对ECG信号应用Pan-Tompkins R波检测算法 
        detectors = Detectors(500)  # 500是采样率
        or_Pan_rpeaks = detectors.pan_tompkins_detector(or_ecg_data)
        de_Pan_rpeaks = detectors.pan_tompkins_detector(de_ecg_data)


        # # 调试时候用的
        # or_Hamilton_rpeaks = true_r_peak_data
        # de_Hamilton_rpeaks = true_r_peak_data
        # or_Pan_rpeaks = true_r_peak_data
        # de_Pan_rpeaks = true_r_peak_data

        
        # 保存R波检测的结果
        scipy.io.savemat(cpsc2020_De_R_mat+data_file_name, {'true_r_peak_data':true_r_peak_data, 'or_Hamilton_rpeaks':or_Hamilton_rpeaks,'de_Hamilton_rpeaks':de_Hamilton_rpeaks \
                        ,'or_Pan_rpeaks':or_Pan_rpeaks, 'de_Pan_rpeaks':de_Pan_rpeaks})


 
"""
    该函数可视化PSC2020数据集去噪前后R波定位情况
"""
cpsc2020_De_R_fig = "./results/cpsc2020_R_result/De_R_fig/"
def CPSC2020_or_de_R_Locate_Plot_save():
    sampling_rate = 500  # 采样率为500Hz
    data_files = [file for file in os.listdir(Denoised_cpsc2020_mat) if file.endswith('.mat')]
    # 循环依次处理每个文件
    print("total files",len(data_files))
    for data_file_name in tqdm(data_files[0:], desc="Processing CPSC2020 R plot ", unit="step"):
        ecg_mat_data = scipy.io.loadmat(Denoised_cpsc2020_mat+data_file_name)
        or_ecg_data = ecg_mat_data['ecg_orl'].flatten()   # 原始数据
        de_ecg_data = ecg_mat_data['ecg_de'].flatten()    # 去噪后的数据

        ecg_rpeaks_data = scipy.io.loadmat(cpsc2020_De_R_mat+data_file_name) # 读取R波标签信息
        true_r_peak_data = ecg_rpeaks_data['true_r_peak_data'].flatten() 
        or_Hamilton_rpeaks = ecg_rpeaks_data['or_Hamilton_rpeaks'].flatten() 
        de_Hamilton_rpeaks = ecg_rpeaks_data['de_Hamilton_rpeaks'].flatten() 
        or_Pan_rpeaks = ecg_rpeaks_data['or_Pan_rpeaks'].flatten() 
        de_Pan_rpeaks = ecg_rpeaks_data['de_Pan_rpeaks'].flatten()

       # 创建存储图片的子文件夹
        fig_subfolder = os.path.join(cpsc2020_De_R_fig, data_file_name[:-4])
        os.makedirs(fig_subfolder, exist_ok=True)
        
        """
            现在开始对没各片段进行绘图处理，绘图并保存绘图的结果、图片可视化按10s一段信号进行可视化
        """
        # 计算每个画图片段的时间范围
        fig_seg_dur = 10  # 每段10秒
        fig_num_segments = len(or_ecg_data) // (500 * fig_seg_dur)  # 一段ecg片段按10s分割，则共有的ECG片段数量
        time_ranges = [(i * fig_seg_dur, (i + 1) * fig_seg_dur) for i in range(fig_num_segments)]
        # 将一大段ecg信号，分段进行绘制图形并保存
        for fig_i, (start_time, end_time) in tqdm(enumerate(time_ranges),desc="R plot ", unit="fig_seg"):
            segment_indices = np.arange(start_time * 500, end_time * 500)   # 获得绘制该段信号的ecg片段x轴信息

            # 原始数据中数据值太小的，就不要画图了，容易卡住
            if(abs(np.mean(or_ecg_data[segment_indices],axis=0)) <= 1e-4):
                continue    

            # 获得绘制该段信号的r波范围点
            plot_true_r_peaks_range = (true_r_peak_data >= segment_indices[0]) & (true_r_peak_data < segment_indices[-1])
            plot_or_Hamilton_rpeaks_range = (or_Hamilton_rpeaks >= segment_indices[0]) & (or_Hamilton_rpeaks < segment_indices[-1])
            plot_de_Hamilton_rpeaks_range = (de_Hamilton_rpeaks >= segment_indices[0]) & (de_Hamilton_rpeaks < segment_indices[-1])
            plot_or_Pan_rpeaks_range = (or_Pan_rpeaks >= segment_indices[0]) & (or_Pan_rpeaks < segment_indices[-1])
            plot_de_Pan_rpeaks_range = (de_Pan_rpeaks >= segment_indices[0]) & (de_Pan_rpeaks < segment_indices[-1])

            # 图片绘制、及R波定位结果可视化
            a=plt.figure()
            a.set_size_inches(12, 10)
            ax=plt.subplot(211)
            major_ticksx = np.arange(0, 10*sampling_rate, 1*sampling_rate)
            minor_ticksx = np.arange(0, 10*sampling_rate, 0.25*sampling_rate)
            max = np.max(or_ecg_data[segment_indices]) 
            min = np.min(or_ecg_data[segment_indices]) 
            delet = max-min
            max = math.ceil(max + delet*0.25)
            min = math.floor(min - delet*0.25)
            major_ticksy = np.arange(min, max,0.3*delet)
            minor_ticksy = np.arange(min, max, 0.075*delet)  
            ax.set_xticks(major_ticksx)
            ax.set_xticks(minor_ticksx, minor=True)          
            ax.set_yticks(major_ticksy)
            ax.set_yticks(minor_ticksy, minor=True)

            plt.plot(segment_indices/sampling_rate,or_ecg_data[segment_indices],linewidth=0.7,color='k')  # 绘制原始数据
            # 绘制原始数据R波定位结果
            plt.scatter(true_r_peak_data[plot_true_r_peaks_range]/sampling_rate, 
                        or_ecg_data[true_r_peak_data[plot_true_r_peaks_range]],
                        color='red', label='True R', marker='x')
            plt.scatter(or_Hamilton_rpeaks[plot_or_Hamilton_rpeaks_range]/sampling_rate, 
                        or_ecg_data[or_Hamilton_rpeaks[plot_or_Hamilton_rpeaks_range]], 
                        color='blue', label='Hamilton R', marker='H')
            plt.scatter(or_Pan_rpeaks[plot_or_Pan_rpeaks_range]/sampling_rate, 
                        or_ecg_data[or_Pan_rpeaks[plot_or_Pan_rpeaks_range]], 
                        color='black', label='Pan-Tompkins R', marker='s')

            plt.legend(loc='upper right')
            ax.grid(which='minor', alpha=0.2,color='r')
            ax.grid(which='major', alpha=0.5,color='r')  
            plt.title(f'Or ECG - {data_file_name[:-4]}-Seg {fig_i + 1}')
            plt.xlabel('Time ', fontsize=13)
            plt.ylabel('Amplitude', fontsize=13)
            ax2=plt.subplot(212, sharex = ax)
            major_ticksx = np.arange(0, 10*sampling_rate,1*sampling_rate )
            minor_ticksx = np.arange(0, 10*sampling_rate, 0.25*sampling_rate)
            max = np.max(de_ecg_data[segment_indices]) 
            min = np.min(de_ecg_data[segment_indices]) 
            delet = max-min
            max = math.ceil(max + delet*0.25)
            min = math.floor(min - delet*0.25)
            major_ticksy = np.arange(min, max,0.3*delet)
            minor_ticksy = np.arange(min, max, 0.075*delet)  
            ax2.set_xticks(major_ticksx)
            ax2.set_xticks(minor_ticksx, minor=True)         
            ax2.set_yticks(major_ticksy)
            ax2.set_yticks(minor_ticksy, minor=True)


            plt.plot(segment_indices/sampling_rate,de_ecg_data[segment_indices], linewidth=0.7,color='k')
            # 绘制抗噪后数据R波定位结果
            plt.scatter(true_r_peak_data[plot_true_r_peaks_range]/sampling_rate, 
                        or_ecg_data[true_r_peak_data[plot_true_r_peaks_range]],
                        color='red', label='True R', marker='x')
            plt.scatter(de_Hamilton_rpeaks[plot_de_Hamilton_rpeaks_range]/sampling_rate, 
                        de_ecg_data[de_Hamilton_rpeaks[plot_de_Hamilton_rpeaks_range]], 
                        color='blue', label='Hamilton R', marker='H')
            plt.scatter(de_Pan_rpeaks[plot_de_Pan_rpeaks_range]/sampling_rate, 
                        de_ecg_data[de_Pan_rpeaks[plot_de_Pan_rpeaks_range]], 
                        color='black', label='Pan-Tompkins R', marker='s')
            plt.legend(loc='upper right')
            ax2.grid(which='minor', alpha=0.2,color='r')
            ax2.grid(which='major', alpha=0.5,color='r')  
            plt.title(f"Denoised ECG {start_time}s——{end_time}s", fontsize=15)
            plt.xlabel('Time', fontsize=13)
            plt.ylabel('Amplitude', fontsize=13)
            fig_name = f"{data_file_name[:-4]}_Seg_{fig_i + 1}.jpg"
            fig_path = os.path.join(fig_subfolder, fig_name)
            plt.savefig(fig_path,dpi=100) # 一般的屏幕显示dip100将足够了、科学出版物用dip600左右比较好  
            plt.close()





"""
    该函数计算CPSC2020数据集去噪前后 R 波定位的性能对比
"""
def CPSC2020_R_Metric_Calcu():
    # 原始的信号使用  定位R波的情况
    or_Ham_R_TP = []
    or_Ham_R_FN = []
    or_Ham_R_FP = []
    or_PanT_R_TP = []
    or_PanT_R_FN = []
    or_PanT_R_FP = []
    # 去噪后的信号使用  定位R波的情况
    de_Ham_R_TP = []
    de_Ham_R_FN = []
    de_Ham_R_FP = []
    de_PanT_R_TP = []
    de_PanT_R_FN = []
    de_PanT_R_FP = []

    data_files = [file for file in os.listdir(cpsc2020_De_R_mat) if file.endswith('.mat')]
    # 循环依次处理每个文件
    print("total files",len(data_files))
    for data_file_name in tqdm(data_files[0:], desc="Processing CPSC2020 Metric_Calcu ", unit="step"):

        ecg_rpeaks_data = scipy.io.loadmat(cpsc2020_De_R_mat+data_file_name) # 读取R波标签信息
        true_r_peak_data = ecg_rpeaks_data['true_r_peak_data'].flatten() 
        or_Hamilton_rpeaks = ecg_rpeaks_data['or_Hamilton_rpeaks'].flatten() 
        de_Hamilton_rpeaks = ecg_rpeaks_data['de_Hamilton_rpeaks'].flatten() 
        or_Pan_rpeaks = ecg_rpeaks_data['or_Pan_rpeaks'].flatten() 
        de_Pan_rpeaks = ecg_rpeaks_data['de_Pan_rpeaks'].flatten()
    
        # 计算去噪前后，R波定位的性能指标
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,or_Hamilton_rpeaks)
        or_Ham_R_TP.append(TP)
        or_Ham_R_FN.append(FN)
        or_Ham_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,de_Hamilton_rpeaks)
        de_Ham_R_TP.append(TP)
        de_Ham_R_FN.append(FN)
        de_Ham_R_FP.append(FP)
        # PT 算法
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,or_Pan_rpeaks)
        or_PanT_R_TP.append(TP)
        or_PanT_R_FN.append(FN)
        or_PanT_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,de_Pan_rpeaks)
        de_PanT_R_TP.append(TP)
        de_PanT_R_FN.append(FN)
        de_PanT_R_FP.append(FP)

        
    # 计算总的性能指标
    or_Hamilton_metrics = calculate_total_metrics(or_Ham_R_TP, or_Ham_R_FN, or_Ham_R_FP)
    de_Hamilton_metrics = calculate_total_metrics(de_Ham_R_TP, de_Ham_R_FN, de_Ham_R_FP)
    or_PanT_metrics = calculate_total_metrics(or_PanT_R_TP, or_PanT_R_FN, or_PanT_R_FP)
    de_PanT_metrics = calculate_total_metrics(de_PanT_R_TP, de_PanT_R_FN, de_PanT_R_FP)

    # 打印结果
    df1 = print_metrics_table("Original Hamilton Metrics", or_Hamilton_metrics)
    df2 = print_metrics_table("Denoised Hamilton Metrics", de_Hamilton_metrics)
    df3 = print_metrics_table("Original PanT Metrics", or_PanT_metrics)
    df4 = print_metrics_table("Denoised PanT Metrics", de_PanT_metrics)

    # 保存识别到的指标为表格
    with pd.ExcelWriter('./metric/CPSC2020_R_Metric_Calcu.xlsx', engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 2) #startrow=df1.shape[0] + 2 意味着将 df2 添加到 Sheet1 的起始行为 df1 的行数加上 2。
        df3.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 4)
        df4.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 6)



Denoised_mit_bih_ar_mat =  "./results/Denoised_mit_bih_ar/mat/"
mit_bih_ar_500hz_ref_path = './Data_set/mit_bih_ar_500hz/ref/'
mit_bih_ar_R_result_path =  './results/mit_bih_ar_R_result/mat/'
"""
    该函数同时使用多种R波定位算法，对 MIT BIH  AR 数据库原始信号和抗噪信号进行R波定位，并记录R波定位结果
"""
def MIT_BIH_AR_or_de_R_Locate():
    # 获取所有的.mat文件
    data_files = [file for file in os.listdir(Denoised_mit_bih_ar_mat) if file.endswith('.mat')]

    # 循环依次处理每个文件
    print("total files",len(data_files))
    for data_file_name in tqdm(data_files[0:], desc="Processing MIT_BIH_AR R_Locate", unit="step"):
        # 读取数据
        ecg_mat_data = scipy.io.loadmat(Denoised_mit_bih_ar_mat+data_file_name)
        mill_orl = ecg_mat_data['mill_orl'].flatten()   # 原始数据
        mill_de = ecg_mat_data['mill_de'].flatten()    # 去噪后的数据
        v1_orl = ecg_mat_data['v1_orl'].flatten()   # 原始数据
        v1_de = ecg_mat_data['v1_de'].flatten()    # 去噪后的数据

        true_r_peak_data = scipy.io.loadmat(mit_bih_ar_500hz_ref_path+data_file_name)['r_peaks'].flatten()   # 真实R波标签数据

        # 对ECG信号应用 Hamilton R波检测算法
        or_mill_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(mill_orl, sampling_rate=500)[0]
        de_mill_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(mill_de, sampling_rate=500)[0]
        or_v1_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(v1_orl, sampling_rate=500)[0]
        de_v1_Hamilton_rpeaks = biosppy.signals.ecg.hamilton_segmenter(v1_de, sampling_rate=500)[0]

        # 对ECG信号应用Pan-Tompkins R波检测算法 
        detectors = Detectors(500)  # 500是采样率
        or_mlii_Pan_rpeaks = detectors.pan_tompkins_detector(mill_orl)
        de_mlii_Pan_rpeaks = detectors.pan_tompkins_detector(mill_de)
        or_v1_Pan_rpeaks = detectors.pan_tompkins_detector(v1_orl)
        de_v1_Pan_rpeaks = detectors.pan_tompkins_detector(v1_de)

        # 保存R波检测的结果
        scipy.io.savemat(mit_bih_ar_R_result_path+data_file_name, {'true_r_peak_data':true_r_peak_data, \
                                                                   'or_mill_Hamilton_rpeaks':or_mill_Hamilton_rpeaks,\
                                                                   'de_mill_Hamilton_rpeaks':de_mill_Hamilton_rpeaks, \
                                                                    'or_v1_Hamilton_rpeaks':or_v1_Hamilton_rpeaks, \
                                                                    'de_v1_Hamilton_rpeaks':de_v1_Hamilton_rpeaks,\
                                                                    'or_mlii_Pan_rpeaks':or_mlii_Pan_rpeaks,\
                                                                    'de_mlii_Pan_rpeaks':de_mlii_Pan_rpeaks,\
                                                                    'or_v1_Pan_rpeaks':or_v1_Pan_rpeaks,\
                                                                    'de_v1_Pan_rpeaks':de_v1_Pan_rpeaks})

        

"""
    该函数计算MIT_BIH_AR_R数据集去噪前后 R 波定位的性能对比
"""
def MIT_BIH_AR_R_Metric_Calcu():
    # 原始的信号使用  定位R波的情况
    or_mlii_Ham_R_TP = []
    or_mlii_Ham_R_FN = []
    or_mlii_Ham_R_FP = []
    or_mlii_PanT_R_TP = []
    or_mlii_PanT_R_FN = []
    or_mlii_PanT_R_FP = []
    # 去噪后的信号使用  定位R波的情况
    de_mlii_Ham_R_TP = []
    de_mlii_Ham_R_FN = []
    de_mlii_Ham_R_FP = []
    de_mlii_PanT_R_TP = []
    de_mlii_PanT_R_FN = []
    de_mlii_PanT_R_FP = []
    # 原始的信号使用  定位R波的情况
    or_v1_Ham_R_TP = []
    or_v1_Ham_R_FN = []
    or_v1_Ham_R_FP = []
    or_v1_PanT_R_TP = []
    or_v1_PanT_R_FN = []
    or_v1_PanT_R_FP = []
    # 去噪后的信号使用  定位R波的情况
    de_v1_Ham_R_TP = []
    de_v1_Ham_R_FN = []
    de_v1_Ham_R_FP = []
    de_v1_PanT_R_TP = []
    de_v1_PanT_R_FN = []
    de_v1_PanT_R_FP = []

    data_files = [file for file in os.listdir(mit_bih_ar_R_result_path) if file.endswith('.mat')]
    # 循环依次处理每个文件
    print("total files",len(data_files))
    for data_file_name in tqdm(data_files[0:], desc="Processing CPSC2020 R Metric_Calcu ", unit="step"):

        ecg_rpeaks_data = scipy.io.loadmat(mit_bih_ar_R_result_path+data_file_name) # 读取R波标签信息
        true_r_peak_data = ecg_rpeaks_data['true_r_peak_data'].flatten() 
        or_mill_Hamilton_rpeaks = ecg_rpeaks_data['or_mill_Hamilton_rpeaks'].flatten() 
        de_mill_Hamilton_rpeaks = ecg_rpeaks_data['de_mill_Hamilton_rpeaks'].flatten() 
        or_v1_Hamilton_rpeaks = ecg_rpeaks_data['or_v1_Hamilton_rpeaks'].flatten() 
        de_v1_Hamilton_rpeaks = ecg_rpeaks_data['de_v1_Hamilton_rpeaks'].flatten()

        or_mlii_Pan_rpeaks = ecg_rpeaks_data['or_mlii_Pan_rpeaks'].flatten() 
        de_mlii_Pan_rpeaks = ecg_rpeaks_data['de_mlii_Pan_rpeaks'].flatten() 
        or_v1_Pan_rpeaks = ecg_rpeaks_data['or_v1_Pan_rpeaks'].flatten() 
        de_v1_Pan_rpeaks = ecg_rpeaks_data['de_v1_Pan_rpeaks'].flatten()
    
        # 计算去噪前后，R波定位的性能指标
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,or_mill_Hamilton_rpeaks)
        or_mlii_Ham_R_TP.append(TP)
        or_mlii_Ham_R_FN.append(FN)
        or_mlii_Ham_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,de_mill_Hamilton_rpeaks)
        de_mlii_Ham_R_TP.append(TP)
        de_mlii_Ham_R_FN.append(FN)
        de_mlii_Ham_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,or_v1_Hamilton_rpeaks)
        or_v1_Ham_R_TP.append(TP)
        or_v1_Ham_R_FN.append(FN)
        or_v1_Ham_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,de_v1_Hamilton_rpeaks)
        de_v1_Ham_R_TP.append(TP)
        de_v1_Ham_R_FN.append(FN)
        de_v1_Ham_R_FP.append(FP)
        # PT 算法
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,or_mlii_Pan_rpeaks)
        or_mlii_PanT_R_TP.append(TP)
        or_mlii_PanT_R_FN.append(FN)
        or_mlii_PanT_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,de_mlii_Pan_rpeaks)
        de_mlii_PanT_R_TP.append(TP)
        de_mlii_PanT_R_FN.append(FN)
        de_mlii_PanT_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,or_v1_Pan_rpeaks)
        or_v1_PanT_R_TP.append(TP)
        or_v1_PanT_R_FN.append(FN)
        or_v1_PanT_R_FP.append(FP)
        TP, FN, FP = calculate_R_metrics(true_r_peak_data,de_v1_Pan_rpeaks)
        de_v1_PanT_R_TP.append(TP)
        de_v1_PanT_R_FN.append(FN)
        de_v1_PanT_R_FP.append(FP)

        
    # 计算总的性能指标
    or_mlii_Hamilton_metrics = calculate_total_metrics(or_mlii_Ham_R_TP, or_mlii_Ham_R_FN, or_mlii_Ham_R_FP)
    de_mlii_Hamilton_metrics = calculate_total_metrics(de_mlii_Ham_R_TP, de_mlii_Ham_R_FN, de_mlii_Ham_R_FP)
    or_v1_Hamilton_metrics = calculate_total_metrics(or_v1_Ham_R_TP, or_v1_Ham_R_FN, or_v1_Ham_R_FP)
    de_v1_Hamilton_metrics = calculate_total_metrics(de_v1_Ham_R_TP, de_v1_Ham_R_FN, de_v1_Ham_R_FP)

    or_mlii_PanT_metrics = calculate_total_metrics(or_mlii_PanT_R_TP, or_mlii_PanT_R_FN, or_mlii_PanT_R_FP)
    de_mlii_PanT_metrics = calculate_total_metrics(de_mlii_PanT_R_TP, de_mlii_PanT_R_FN, de_mlii_PanT_R_FP)
    or_v1_PanT_metrics = calculate_total_metrics(or_v1_PanT_R_TP, or_v1_PanT_R_FN, or_v1_PanT_R_FP)
    de_v1_PanT_metrics = calculate_total_metrics(de_v1_PanT_R_TP, de_v1_PanT_R_FN, de_v1_PanT_R_FP)

    # 打印结果
    df1 = print_metrics_table("Original MLII Hamilton Metrics", or_mlii_Hamilton_metrics)
    df2 = print_metrics_table("Denoised MLII Hamilton Metrics", de_mlii_Hamilton_metrics)
    df3 = print_metrics_table("Original V1 Hamilton Metrics", or_v1_Hamilton_metrics)
    df4 = print_metrics_table("Denoised V1 Hamilton Metrics", de_v1_Hamilton_metrics)
    df5 = print_metrics_table("Original MLII PanT Metrics", or_mlii_PanT_metrics)
    df6 = print_metrics_table("Denoised MLII PanT Metrics", de_mlii_PanT_metrics)
    df7 = print_metrics_table("Original V1 PanT Metrics", or_v1_PanT_metrics)
    df8 = print_metrics_table("Denoised V1 PanT Metrics", de_v1_PanT_metrics)

    # 保存识别到的指标为表格
    with pd.ExcelWriter('./metric/MIT_BIH_AR_R_Metric_Calcu.xlsx', engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 2) #startrow=df1.shape[0] + 2 意味着将 df2 添加到 Sheet1 的起始行为 df1 的行数加上 2。
        df3.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 4)
        df4.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 6)
        df5.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 8)
        df6.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 10)
        df7.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 12)
        df8.to_excel(writer, sheet_name='Sheet1', index=False, startrow=df1.shape[0] + 14)




if __name__ == "__main__":
    # print(len(mitbih_arryth_ecg_names))
    # data = mitbih_arryth_read(mitbih_arryth_ecg_names[0])

    # CPSC2019_R_Visible()
    # CPSC2019_R_locate()
    # CPSC2019_or_de_R_Locate()
    CPSC2019_R_Metric_Calcu()
    # CPSC2020_or_de_R_Locate()
    # CPSC2020_or_de_R_Locate_Plot_save()
    # CPSC2020_R_Metric_Calcu()
    # MIT_BIH_AR_or_de_R_Locate()
    # MIT_BIH_AR_R_Metric_Calcu()

    pass





