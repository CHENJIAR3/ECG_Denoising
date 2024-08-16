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
from Denoising import Denoiser
# 忽略所有警告
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
strategy = tf.distribute.MirroredStrategy()


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
    ecg_dir = "/data/chenjiarong/vitaldb_genecg/"
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
