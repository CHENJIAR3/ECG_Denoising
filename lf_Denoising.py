# 用于各个数据集的ecg信号去噪操作
# author: Linfei



import os
import math
from tqdm import tqdm
import scipy.io
import numpy as np
from  utils import args
from Denoising import Denoiser
import tensorflow as tf
from model_structure import *
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'









"""
    对CPSC2019的10s数据片段进行去噪，并保存去噪结果的函数
"""

# 相关文件路径
modelpath = "./Denoiser"
cpsc2019_data_path = "./Data_set/cpsc2019/train/data/"
Denoised_cpsc2019_mat =  "./results/Denoised_cpsc2019/mat/"
Denoised_cpsc2019_fig =  "./results/Denoised_cpsc2019/fig/"

def CPSC2019_denoiser():
    Denoiser_main=Denoiser(modelpath)
    model = tf.keras.models.load_model(modelpath, compile=False)
    forward_noiser=ForwardDiffusion(args.time_steps)
    alphas = forward_noiser.alphas
    betas = forward_noiser.betas
    alpha_hats =forward_noiser.alpha_hat
    
    # 获取所有CPSC2019的.mat文件
    data_files = [file for file in os.listdir(cpsc2019_data_path) if file.endswith('.mat')]
    sampling_rate = 500  # 采样率为500Hz


    # 循环依次处理每个文件
    print("total files",len(data_files))
    for data_file_name in tqdm(data_files[:], desc="Processing Files", unit="step"):
        ref_file_name = 'R'+data_file_name[4:]
        # 读取数据
        ecg_data = scipy.io.loadmat(cpsc2019_data_path+data_file_name)['ecg']

        # cpsc2019 都是10s的片段、所以这里可加可不加
        Signal_Length = len(ecg_data) 
        if Signal_Length<args.ecglen:
            continue
        
        Zero_Length = args.ecglen-Signal_Length%args.ecglen  # 计算将数据补足到1024的整数倍，应该填充的长度。

        ecgdata_paddings = np.concatenate([ecg_data[:], ecg_data[-Zero_Length:]], axis=0)
        ecgdata_reshaped = ecgdata_paddings.reshape(-1,args.ecglen)  # 将数据reshape为 N x 1024


        ecgdata_reshaped = np.expand_dims(ecgdata_reshaped,axis=-1)  
        ecgdata_reshaped = (ecgdata_reshaped - np.mean(ecgdata_reshaped, axis=1)[:, None]) / np.std(ecgdata_reshaped, axis=1)[:,None] # 对数据进行标准化处理
        

        # 转换数据格式并进行抗噪处理
        ecgdata_reshaped = tf.cast(ecgdata_reshaped,dtype=tf.float32)
        denoised_ecgdata = Denoiser_main.get_denoised(ecgdata_reshaped)
        denoised_ecgdata = np.asarray(denoised_ecgdata).reshape(1,-1)
        denoised_ecgdata = denoised_ecgdata[:,:Signal_Length]


        # 得到抗噪后信号平滑后的信号
        window_size = 250
        smoothed_signal = np.zeros_like((denoised_ecgdata))
        for i in range(denoised_ecgdata.shape[0]):
            smoothed_signal[i,:] = np.convolve(denoised_ecgdata [i,:], np.ones(window_size) / window_size, mode='same')

        denoised_ecgdata=denoised_ecgdata-smoothed_signal
        denoised_ecgdata = denoised_ecgdata.T  # 再转置回去，确保shape和输入shape一样
        # 保存抗噪后的数据为mat文件
        scipy.io.savemat(Denoised_cpsc2019_mat+data_file_name, {'ecg_orl': ecg_data,'ecg_de':denoised_ecgdata})


        # 图片绘制、及抗噪后数据存储
        a=plt.figure()
        a.set_size_inches(12, 10)
        ax=plt.subplot(211)
        major_ticksx = np.arange(0, 10*sampling_rate, 1*sampling_rate)
        minor_ticksx = np.arange(0, 10*sampling_rate, 0.25*sampling_rate)

        max = np.max(ecg_data) 
        min = np.min(ecg_data) 
        delet = max-min
        max = math.ceil(max + delet*0.25)
        min = math.floor(min - delet*0.25)

        major_ticksy = np.arange(min, max,0.3*delet)
        minor_ticksy = np.arange(min, max, 0.075*delet)  
        ax.set_xticks(major_ticksx)
        ax.set_xticks(minor_ticksx, minor=True)          
        ax.set_yticks(major_ticksy)
        ax.set_yticks(minor_ticksy, minor=True)
        plt.plot(ecg_data,linewidth=0.7,color='k')
        ax.grid(which='minor', alpha=0.2,color='r')
        ax.grid(which='major', alpha=0.5,color='r')  
        plt.title("Original ECG", fontsize=15)
        # plt.axis([0, 10,-1.5, 1.5])
        plt.xlabel('Time ', fontsize=13)
        plt.ylabel('Amplitude', fontsize=13)

        ax2=plt.subplot(212, sharex = ax)
        major_ticksx = np.arange(0, 10*sampling_rate,1*sampling_rate )
        minor_ticksx = np.arange(0, 10*sampling_rate, 0.25*sampling_rate)
        max = np.max(denoised_ecgdata) 
        min = np.min(denoised_ecgdata) 
        delet = max-min
        max = math.ceil(max + delet*0.25)
        min = math.floor(min - delet*0.25)
        major_ticksy = np.arange(min, max,0.3*delet)
        minor_ticksy = np.arange(min, max, 0.075*delet)  

        ax2.set_xticks(major_ticksx)
        ax2.set_xticks(minor_ticksx, minor=True)         
        ax2.set_yticks(major_ticksy)
        ax2.set_yticks(minor_ticksy, minor=True)
        plt.plot(denoised_ecgdata, linewidth=0.7,color='k')
        ax2.grid(which='minor', alpha=0.2,color='r')
        ax2.grid(which='major', alpha=0.5,color='r')  
        plt.title("Denoised ECG", fontsize=15)
        # plt.axis([0, 10,-1.5, 1.5])
        plt.xlabel('Time', fontsize=13)
        plt.ylabel('Amplitude', fontsize=13)
        plt.savefig(Denoised_cpsc2019_fig+data_file_name[:-4]+'.jpg',dpi=100) # 一般的屏幕显示dip100将足够了、科学出版物用dip600左右比较好  
        plt.close()

    print("Finishing...")



# 相关文件路径
modelpath = "./Denoiser"

cpsc2020_500hz_data_path = "./Data_set/cpsc2020_500hz/data/"

Denoised_cpsc2020_mat =  "./results/Denoised_cpsc2020/mat/"
Denoised_cpsc2020_fig =  "./results/Denoised_cpsc2020/fig/"


"""
    对CPSC2020的数据集进行去噪，并保存去噪结果的函数 
"""
def CPSC2020_denoiser():
    Denoiser_main=Denoiser(modelpath)
    model = tf.keras.models.load_model(modelpath, compile=False)
    forward_noiser=ForwardDiffusion(args.time_steps)
    alphas = forward_noiser.alphas
    betas = forward_noiser.betas
    alpha_hats =forward_noiser.alpha_hat

    # 获取所有的心电数据.mat文件
    data_files = [file for file in os.listdir(cpsc2020_500hz_data_path) if file.endswith('.mat')]
    for data_file_name in tqdm(data_files[0:], desc="Processing Files", unit="step"):
        # 读取心电信号
        data_path = os.path.join(cpsc2020_500hz_data_path, data_file_name)
        ecg_mat_data = scipy.io.loadmat(data_path)
        all_ecg_data = ecg_mat_data['ecg'].flatten()  # 一段ECG数据太长了，
        # 创建存储图片的子文件夹
        fig_subfolder = os.path.join(Denoised_cpsc2020_fig, data_file_name[:-4])
        os.makedirs(fig_subfolder, exist_ok=True)

        # 按5分钟进行一段段处理
        ecg_segment_duration = 60*5
        # 计算可以分割的段数
        total_duration = len(all_ecg_data)
        num_segments = int(np.ceil(total_duration / (500*ecg_segment_duration)))

        all_denoised_ecgdata = []
        # 一段段遍历
        for seg_i in tqdm(range(num_segments), desc=f"Processing Segments - {data_file_name[:-4]}", unit="segment"):
        
            # 计算当前段的起始和结束索引
            start_index = seg_i * int(ecg_segment_duration * 500)
            end_index = np.min([(seg_i + 1)*int(ecg_segment_duration*500), len(all_ecg_data)])
            ecg_data = all_ecg_data[start_index:end_index] # 分割为片段后再处理、一小段一小段处理

            Signal_Length = len(ecg_data) 
            if Signal_Length<args.ecglen:
                continue
            
            Zero_Length = args.ecglen-Signal_Length%args.ecglen  # 计算将数据补足到1024的整数倍，应该填充的长度。
            ecgdata_paddings = np.concatenate([ecg_data[:], ecg_data[-Zero_Length:]], axis=0)
            ecgdata_reshaped = ecgdata_paddings.reshape(-1,args.ecglen)  # 将数据reshape为 N x 1024
            ecgdata_reshaped = np.expand_dims(ecgdata_reshaped,axis=-1)  
            ecgdata_reshaped = (ecgdata_reshaped - np.mean(ecgdata_reshaped, axis=1)[:, None]) / np.std(ecgdata_reshaped, axis=1)[:,None] # 对数据进行标准化处理
            # 转换数据格式并进行抗噪处理
            ecgdata_reshaped = tf.cast(ecgdata_reshaped,dtype=tf.float32)
            denoised_ecgdata = Denoiser_main.get_denoised(ecgdata_reshaped)
            denoised_ecgdata = np.asarray(denoised_ecgdata).reshape(1,-1)
            denoised_ecgdata = denoised_ecgdata[:,:Signal_Length]
            # 得到抗噪后信号平滑后的信号
            window_size = 250
            smoothed_signal = np.zeros_like((denoised_ecgdata))
            for i in range(denoised_ecgdata.shape[0]):
                smoothed_signal[i,:] = np.convolve(denoised_ecgdata [i,:], np.ones(window_size) / window_size, mode='same')
            denoised_ecgdata=denoised_ecgdata-smoothed_signal
            denoised_ecgdata = denoised_ecgdata.T  # 再转置回去，确保shape和输入shape一样


            # 将处理好的段添加到列表中
            all_denoised_ecgdata.append(denoised_ecgdata.flatten())       


            """
                现在开始对没各片段进行绘图处理，绘图并保存绘图的结果、图片可视化按10s一段信号进行可视化
            """
            # 计算每个画图片段的时间范围
            fig_seg_dur = 10  # 每段10秒
            fig_num_segments = len(denoised_ecgdata) // (500 * fig_seg_dur)  # 一段ecg片段按10s分割，则共有的ECG片段数量

            time_ranges = [(i * fig_seg_dur, (i + 1) * fig_seg_dur) for i in range(fig_num_segments)]
            # 绘制图形并保存
            for fig_i, (start_time, end_time) in enumerate(time_ranges):
                segment_indices = np.arange(start_time * 500, end_time * 500)

                sampling_rate = 500
                a=plt.figure()
                a.set_size_inches(12, 10)
                ax=plt.subplot(211)
                major_ticksx = np.arange(0, 10*sampling_rate, 1*sampling_rate)
                minor_ticksx = np.arange(0, 10*sampling_rate, 0.25*sampling_rate)
                max = np.max(ecg_data) 
                min = np.min(ecg_data) 
                delet = max-min
                max = math.ceil(max + delet*0.25)
                min = math.floor(min - delet*0.25)
                major_ticksy = np.arange(min, max,0.3*delet)
                minor_ticksy = np.arange(min, max, 0.075*delet)  
                ax.set_xticks(major_ticksx)
                ax.set_xticks(minor_ticksx, minor=True)          
                ax.set_yticks(major_ticksy)
                ax.set_yticks(minor_ticksy, minor=True)
                plt.plot(segment_indices/sampling_rate,ecg_data[segment_indices],linewidth=0.7,color='k')
                ax.grid(which='minor', alpha=0.2,color='r')
                ax.grid(which='major', alpha=0.5,color='r')  
                plt.title(f"Original ECG Signal - {data_file_name[:-4]}-Block {seg_i} -Segment {fig_i}", fontsize=15)
                plt.xlabel('Time ', fontsize=13)
                plt.ylabel('Amplitude', fontsize=13)
 

                ax2=plt.subplot(212, sharex = ax)
                major_ticksx = np.arange(0, 10*sampling_rate,1*sampling_rate )
                minor_ticksx = np.arange(0, 10*sampling_rate, 0.25*sampling_rate)
                max = np.max(denoised_ecgdata) 
                min = np.min(denoised_ecgdata) 
                delet = max-min
                max = math.ceil(max + delet*0.25)
                min = math.floor(min - delet*0.25)
                major_ticksy = np.arange(min, max,0.3*delet)
                minor_ticksy = np.arange(min, max, 0.075*delet)  
                ax2.set_xticks(major_ticksx)
                ax2.set_xticks(minor_ticksx, minor=True)         
                ax2.set_yticks(major_ticksy)
                ax2.set_yticks(minor_ticksy, minor=True)
                plt.plot(segment_indices/sampling_rate,denoised_ecgdata[segment_indices], linewidth=0.7,color='k')
                ax2.grid(which='minor', alpha=0.2,color='r')
                ax2.grid(which='major', alpha=0.5,color='r')  
                plt.title("Denoised ECG", fontsize=15)
                plt.xlabel('Time', fontsize=13)
                plt.ylabel('Amplitude', fontsize=13)
                plt.xticks()
                plt.legend()
                plt.grid(True)
                plt.show()
                plt.savefig(fig_subfolder+'/'+data_file_name[:-4]+'_'+str(seg_i)+'_'+str(fig_i)+'.jpg',dpi=100) # 一般的屏幕显示dip100将足够了、科学出版物用dip600左右比较好  
                plt.close()

        # 最后将所有段拼接成一个大的数组
        all_denoised_ecgdata = np.concatenate(all_denoised_ecgdata, axis=0)
        # 保存抗噪后的数据为mat文件
        scipy.io.savemat(Denoised_cpsc2020_mat+data_file_name, {'ecg_orl': all_ecg_data,'ecg_de':all_denoised_ecgdata})




if __name__=='__main__':
    CPSC2020_denoiser()
    pass




