import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import matplotlib.pyplot as plt
# 把单导联心电图作为输入信号
from model_structure import *

import tensorflow as tf
from utils import args
import warnings
import glob
import scipy.io as sio
from model_structure import unet3plus
import pandas as pd
from scipy.signal import lfilter, butter
from Denoising import Denoiser

# 忽略所有警告
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
strategy = tf.distribute.MirroredStrategy()


def ecg_processing(data):
    # 这里可以补零,也可以采取别的思路
    Signal_Length = data.shape[0]
    Signal_ch = data.shape[1]
    Pad_Length = args.ecglen - Signal_Length % args.ecglen

    data_paddings = np.concatenate([data[:Signal_Length, :], data[-Pad_Length:, :]], axis=0)
    data_reshaped = data_paddings.reshape(args.ecglen, -1, order='F').transpose()
    data_reshaped = (data_reshaped - np.mean(data_reshaped, axis=1)[:, None]) / np.std(data_reshaped, axis=1)[:, None]
    data_reshaped = tf.cast(tf.expand_dims(data_reshaped, axis=-1), dtype=tf.float32)
    for rt in range(args.Repeat_time):
        denoised_data = Denoiser_main.get_denoised(data_reshaped)
    denoised_data = np.asarray(denoised_data).transpose().reshape(-1, Signal_ch, order='F')
    denoised_data = denoised_data[:Signal_Length, :]
    window_size = 250
    smoothed_signal = np.zeros_like(denoised_data)
    for i in range(Signal_ch):
        smoothed_signal[:, i] = np.convolve(denoised_data[:, i], np.ones(window_size) / window_size, mode='same')
    return data, denoised_data - smoothed_signal


if __name__ == '__main__':
    modelpath = "./Denoiser"
    args.Repeat_time = 2
    txtpaths = glob.glob("/home/chenjiarong/SingleArm_ECGDenoising/single_arm/txtdata/*.txt")
    Denoiser_main = Denoiser(modelpath)
    for txtpath in txtpaths:
        data = pd.read_csv(txtpath, header=0, names=['ch1', 'ch2'])
        data_vals = data.values
        args.Repeat_time = 1
        data_vals, denoised_vals = ecg_processing(data_vals)

        plt.figure()
        plt.subplot(411)
        Sig_len = min(data_vals.shape[0], 10 * args.fs)
        t = np.arange(Sig_len) / args.fs

        plt.plot(t, data_vals[-Sig_len:, 0], 'r', label="Noised CH1")
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(412)
        plt.plot(t, denoised_vals[-Sig_len:, 0], 'b', label="Denoised CH1")
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(413)
        plt.plot(t, data_vals[-Sig_len:, 1], 'r', label="Noised CH2")
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(414)
        plt.plot(t, denoised_vals[-Sig_len:, 1], 'b', label="Denoised CH2")
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.grid(True)
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig('./single_arm/txtdata/' + txtpath.split('/')[-1][:-4] + '_Denoised', dpi=600)
        #
        sio.savemat('./single_arm/txtdata/' + txtpath.split('/')[-1][:-4] + '.mat',
                    {'Original Input': data_vals, 'Denoised Output': denoised_vals})
