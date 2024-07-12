# 去噪函数

import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import matplotlib.pyplot as plt
# 把单导联心电图作为输入信号
from model_structure import *
import pickle
import tensorflow as tf
from  utils import args
import warnings
import glob
import scipy.io as sio
from model_structure import unet3plus

# 忽略所有警告
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
strategy = tf.distribute.MirroredStrategy()

class Denoiser:
    def __init__(self,modelpath= "./Denoiser"):
        self.modelpath=modelpath
        self.model = tf.keras.models.load_model(self.modelpath)
        self.time_steps = args.time_steps
        self.forward_noiser = ForwardDiffusion(args.time_steps)
        self.alphas = self.forward_noiser.alphas
        self.betas = self.forward_noiser.betas
        self.alpha_hats = self.forward_noiser.alpha_hat
        self.Repeat_time=1
    def get_denoised(self,xt):
        # xt = (xt - tf.reduce_mean(xt,axis=1,keepdims=True))/(tf.math.reduce_std(xt,axis=1,keepdims=True))
        for rt in range(self.Repeat_time):
            # time_steps= int(self.time_steps*(1-rt/(2*self.Repeat_time)))
            for t in reversed(range(1,self.time_steps)):
                time = tf.repeat(tf.constant(t, dtype=tf.int32), repeats=xt.shape[0], axis=0)
                time = tf.expand_dims(time, axis=-1)
                predicted_noise = self.model([xt, time], training=False)
                alpha = tf.gather(self.alphas, time)[:, None]
                alpha_hat = tf.gather(self.alpha_hats, time)[:, None]
                alpha_hat_t_1 = tf.gather(self.alpha_hats, time - 1)[:, None]

                if  time[0] > 1:
                    noise = tf.cast((tf.random.normal(shape=tf.shape(xt))), dtype=tf.float32)
                else:
                    noise = tf.cast(tf.zeros_like(xt), dtype=tf.float32)
                xt = (1 / tf.sqrt(alpha)) * (xt - ((1 - alpha) / (tf.sqrt(1 - alpha_hat))) * predicted_noise) + tf.sqrt(
                    (1 - alpha) * (1 - alpha_hat_t_1) / (1 - alpha_hat)) * noise
            # xt = (xt - tf.reduce_mean(xt,axis=1,keepdims=True))/(tf.math.reduce_std(xt,axis=1,keepdims=True))
        return xt
def Denoising_ECG(Denoiser,ecgdata,fs=500):
    ecgdata = np.array([[num for num in sublist] for sublist in ecgdata])
    if fs!=500:
        # 通常为1*N
        from scipy.interpolate import interp1d
        f = interp1d(np.arange(len(ecgdata[0])), ecgdata, kind='cubic', fill_value="extrapolate")
        # 上采样
        ecgdata = f(np.linspace(0, len(ecgdata[0]), int(len(ecgdata[0]) *500/ fs)))

    Signal_Length = ecgdata.shape[1]
    Zero_Length = args.ecglen - Signal_Length % args.ecglen
    ecgdata_paddings = np.concatenate([ecgdata[:, :Signal_Length], ecgdata[:, -Zero_Length:]], axis=1)
    ecgdata_reshaped = ecgdata_paddings.reshape(-1, args.ecglen)
    ecgdata_reshaped = np.expand_dims(ecgdata_reshaped, axis=-1)
    
    ecgdata_reshaped = tf.cast(ecgdata_reshaped, dtype=tf.float32)
    denoised_ecgdata = Denoiser.get_denoised(ecgdata_reshaped)
    denoised_ecgdata = np.asarray(denoised_ecgdata).reshape(-1, Signal_Length+Zero_Length)
    denoised_ecgdata = denoised_ecgdata[:, :Signal_Length]
    if fs!=500:
        # 通常为1*N
        from scipy.interpolate import interp1d
        f2 = interp1d(np.arange(len(denoised_ecgdata[0])), denoised_ecgdata, kind='cubic', fill_value="extrapolate")
        # 上采样
        denoised_ecgdata = f2(np.linspace(0, len(denoised_ecgdata[0]), int(len(denoised_ecgdata[0]) *fs/500)))
    return denoised_ecgdata

def detrend_ECG(denoised_ecgdata,window_size = 250):

    smoothed_signal = np.zeros_like((denoised_ecgdata))
    for i in range(denoised_ecgdata.shape[0]):
        smoothed_signal[i,:] = np.convolve(denoised_ecgdata [i,:], np.ones(window_size) / window_size, mode='same')
    denoised_ecgdata=denoised_ecgdata-smoothed_signal
    return denoised_ecgdata

if __name__=='__main__':
    args.samplemethod = 'ddpm'
    modelpath = "./Denoiser"
    Denoiser_1=Denoiser(modelpath)

    # matpath='./results/Denoised_data/'
    # figpath='./results/Figures/'
    # if not os.path.exists(matpath):
    #     os.mkdir(matpath)
    # if not os.path.exists(figpath):
    #     os.mkdir(figpath)

    model = tf.keras.models.load_model(modelpath)
    forward_noiser=ForwardDiffusion(args.time_steps)
    alphas = forward_noiser.alphas
    betas = forward_noiser.betas
    alpha_hats =forward_noiser.alpha_hat
    gecg_dir = "/data/chenjiarong/vitaldb_genecg/"
    deecg_dir = "/data/chenjiarong/vitaldb_genecg_df/"
    if not os.path.exists(deecg_dir):
        os.mkdir(deecg_dir)
    pkl_files = glob.glob(gecg_dir+"*.pkl")
    for path in pkl_files:
        # pass
        print(path)
        with open(path, 'rb') as file:
            data = pickle.load(file)
        denoised_ecg=Denoising_ECG(Denoiser_1,data["gen_ecg"],100)
        print(np.mean(denoised_ecg),denoised_ecg.shape)
        datanew = data.assign(denoised_ecg=denoised_ecg.tolist())

        newpath=deecg_dir+path.split('.')[0].split('/')[-1]+"&gen_ecg_df.pkl"
        with open(newpath, 'wb') as file:
            pickle.dump(datanew,file)