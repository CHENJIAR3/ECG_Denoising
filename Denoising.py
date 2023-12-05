# 去噪函数

import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
import matplotlib.pyplot as plt
# 把单导联心电图作为输入信号
from model_structure import *

import tensorflow as tf
from utils import args
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
    def __init__(self, modelpath="./Denoiser"):
        self.modelpath = modelpath
        self.model = tf.keras.models.load_model(self.modelpath)
        self.time_steps = args.time_steps
        self.forward_noiser = ForwardDiffusion(args.time_steps)
        self.alphas = self.forward_noiser.alphas
        self.betas = self.forward_noiser.betas
        self.alpha_hats = self.forward_noiser.alpha_hat
        self.Repeat_time = 2

    def get_denoised(self, xt):
        for rt in range(self.Repeat_time):
            time_steps = int(self.time_steps * (1 - rt / (2 * self.Repeat_time)))
            for t in reversed(range(1, time_steps)):
                time = tf.repeat(tf.constant(t, dtype=tf.int32), repeats=xt.shape[0], axis=0)
                time = tf.expand_dims(time, axis=-1)
                predicted_noise = self.model([xt, time], training=False)
                alpha = tf.gather(self.alphas, time)[:, None]
                alpha_hat = tf.gather(self.alpha_hats, time)[:, None]
                alpha_hat_t_1 = tf.gather(self.alpha_hats, time - 1)[:, None]

                if time[0] > 1:
                    noise = tf.cast((tf.random.normal(shape=tf.shape(xt))), dtype=tf.float32)
                else:
                    noise = tf.cast(tf.zeros_like(xt), dtype=tf.float32)
                xt = (1 / tf.sqrt(alpha)) * (xt - ((1 - alpha) / (tf.sqrt(1 - alpha_hat))) * predicted_noise) + tf.sqrt(
                    (1 - alpha) * (1 - alpha_hat_t_1) / (1 - alpha_hat)) * noise
            xt = (xt - tf.reduce_mean(xt, axis=1, keepdims=True)) / (tf.math.reduce_std(xt, axis=1, keepdims=True))
        return xt


if __name__ == '__main__':
    args.samplemethod = 'ddpm'
    modelpath = "./Denoiser"
    matpath = './results/Denoised_data/'
    figpath = './results/Figures/'
    Denoiser_main = Denoiser(modelpath)
    if not os.path.exists(matpath):
        os.mkdir(matpath)
    if not os.path.exists(figpath):
        os.mkdir(figpath)

    model = tf.keras.models.load_model(modelpath)
    forward_noiser = ForwardDiffusion(args.time_steps)
    alphas = forward_noiser.alphas
    betas = forward_noiser.betas
    alpha_hats = forward_noiser.alpha_hat

    # 需要处理的数据
    target_path = '/home/chenjiarong/SingleArm_ECGDenoising/single_arm/Device_Data2/'
    ecgpaths = glob.glob(target_path + "*.mat")
    datas = [sio.loadmat(f)['mydata'] for f in ecgpaths]
    idx = 0
    for ecgdata in datas[idx:]:
        idx = idx + 1

        Signal_Length = ecgdata.shape[1]
        if Signal_Length < args.ecglen:
            continue
        # 导联脱落
        # mean_val=np.mean(ecgdata)
        # index = ecgdata[0] > 3 * mean_val
        # ecgdata[0, index] = mean_val
        Zero_Length = args.ecglen - Signal_Length % args.ecglen

        ecgdata_paddings = np.concatenate([ecgdata[:, :Signal_Length], ecgdata[:, -Zero_Length:]], axis=1)
        ecgdata_reshaped = ecgdata_paddings.reshape(-1, args.ecglen)
        ecgdata_reshaped = np.expand_dims(ecgdata_reshaped, axis=-1)
        ecgdata_reshaped = (ecgdata_reshaped - np.mean(ecgdata_reshaped, axis=1)[:, None]) / np.std(ecgdata_reshaped,
                                                                                                    axis=1)[:,
                                                                                             None]

        ecgdata_reshaped = tf.cast(ecgdata_reshaped, dtype=tf.float32)
        denoised_ecgdata = Denoiser_main.get_denoised(ecgdata_reshaped)
        denoised_ecgdata = np.asarray(denoised_ecgdata).reshape(1, -1)
        denoised_ecgdata = denoised_ecgdata[:, :Signal_Length]

        window_size = 250
        smoothed_signal = np.zeros_like((denoised_ecgdata))
        for i in range(denoised_ecgdata.shape[0]):
            smoothed_signal[i, :] = np.convolve(denoised_ecgdata[i, :], np.ones(window_size) / window_size, mode='same')
        denoised_ecgdata = denoised_ecgdata - smoothed_signal

        args.max_Len = 5 * args.fs
        if Signal_Length > args.max_Len:
            Signal_Length = args.max_Len
        t = np.arange(Signal_Length) / args.fs
        rect = np.array([0.1, 0.55, 0.8, 0.35])
        plt.figure()
        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.plot(t, ecgdata[0, :Signal_Length], color='b', label='Noised Input')
        plt.legend(loc='upper right')
        plt.grid(True)
        rect[1] -= 0.45
        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.plot(t, denoised_ecgdata[0, :Signal_Length], color='k', label='Denoised Output')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.savefig(figpath + 'Denoised_' + str(idx), dpi=600)
        sio.savemat(matpath + 'data' + str(idx) + '.mat',
                    {'Original Input': ecgdata, 'Denoised Output': denoised_ecgdata})
        print("Finishing...")
