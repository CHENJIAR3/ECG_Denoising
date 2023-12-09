# This file is used to calculate the change of R-wave calculation accuracy before and after signal denoising
import os
import wfdb
import numpy as np
import scipy.io as scio
import scipy.io
import matplotlib.pyplot as plt

"""
    cpsc2018  500Hz采样率 12导联心律失常分类任务
    Normal	AF	I-AVB	LBBB	RBBB	PAC	PVC	STD	STE
    1       2    3      4        5      6    7   8   9
    http://2018.icbeb.org/Challenge.html

"""

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



class mitbih_arryth_read():
    def __init__(self, num, path=None) -> None:
        if not path:
            if os.path.exists("../Data_set/mit-bih-arrhythmia-database-1.0.0/"):
                path = "../Data_set/mit-bih-arrhythmia-database-1.0.0/"
            elif os.path.exists("./Data_set/mit-bih-arrhythmia-database-1.0.0/"):
                path = "./Data_set/mit-bih-arrhythmia-database-1.0.0/"
            else:
                print("Could not find path")
                return None
        self.data = (wfdb.rdrecord(os.path.join(path, str(num)), physical=False)).d_signal
        self.data = np.array(self.data)
        self.ann = wfdb.rdann(os.path.join(path, str(num)), extension="atr").__dict__
        
    def ann(self,):
        return self.ann
    
    def data(self,):
        return self.data
    
    def r_index(self,):
        return self.ann['sample']
    
    def r_type(self,):
        return  self.ann['symbol']
    





"""
    cpsc2019 R波识别任务
    http://2019.icbeb.org/Challenge.html

"""

cpsc2019_data_path = "./Data_set/cpsc2019/train/data/"
cpsc2019_ref_path = "./Data_set/cpsc2019/train/ref/"
cpsc2019_R_figpath = "./results/cpsc2019_R_result/fig/"

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












if __name__ == "__main__":
    # print(len(mitbih_arryth_ecg_names))
    # data = mitbih_arryth_read(mitbih_arryth_ecg_names[0])

    CPSC2019R_Visible()
    pass



