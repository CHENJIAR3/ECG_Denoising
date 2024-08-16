# 去噪函数
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import matplotlib.pyplot as plt

# 把单导联心电图作为输入信号
import pickle
import tensorflow as tf
from  utils import args
import warnings
import glob
import scipy.io as sio
from model_structure import unet3plus
from Denoising import *
# 忽略所有警告
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
def Read_ECG_sample(datapath,lead_idx=0):

    ecg1_all = []
    ecgpaths = glob.glob(datapath + "*.mat")
    for ecgpath in ecgpaths:
        # 读取数据
        '''
        下面读取的是cpsc2018的数据集，并选择I导联，输入数据应该为(N,1024,12)
        '''
        ecg = sio.loadmat(ecgpath)['ECG'][0][0][2]
        ecg1 = ecg[lead_idx,:]
        ecg1_all.append(ecg1)

    return ecg1_all
if __name__=='__main__':
    args.samplemethod = 'ddpm'
    modelpath = "./Denoiser"
    fs = 500
    datapath = './Sample_data/'
    Denoiser_1=Denoiser(modelpath)
    try:
        model = tf.keras.models.load_model(modelpath)
        original_ecg = Read_ECG_sample(datapath=datapath)
        denoised_ecg=Denoising_ECG(Denoiser_1,original_ecg,fs)
    except:
        pass
