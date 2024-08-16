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

if __name__=='__main__':
    args.samplemethod = 'ddpm'
    modelpath = "./Denoiser"
    fs = 100
    Denoiser_1=Denoiser(modelpath)

    model = tf.keras.models.load_model(modelpath)
    forward_noiser=ForwardDiffusion(args.time_steps)
    alphas = forward_noiser.alphas
    betas = forward_noiser.betas
    alpha_hats =forward_noiser.alpha_hat
    ecg_dir = "original_ecg_path"
    deecg_dir = "denoised_ecg_path"
    if not os.path.exists(deecg_dir):
        os.mkdir(deecg_dir)
    pkl_files = glob.glob(gecg_dir+"*.pkl")
    for path in pkl_files:
        # pass
        print(path)
        with open(path, 'rb') as file:
            data = pickle.load(file)
        denoised_ecg=Denoising_ECG(Denoiser_1,data["gen_ecg"],fs)
