# 这个函数用于制作数据集，其中采用CPSC2018的一导联数据集和实际采集的单臂心电图
import numpy as np
import pickle
import glob
import scipy.io as sio
from utils import args
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import tensorflow as tf
import os
# 加载信号
def load_raw_data(datapath):
    #
    ecgpaths =glob.glob(datapath+"*.mat")
    datas = [sio.loadmat(f)['mydata'] for f in ecgpaths]
    beat_all=[]
    nodata=1
    i=0
    for data in datas:
        print(i)
        if len(data.reshape(-1))<args.ecglen:
            continue
        i=i+1
        Sig_Len = int((data.shape[1] // args.ecglen) * args.ecglen)
        rdata = data[:, :Sig_Len]
        beat = rdata.reshape(-1, args.ecglen, 1)
        beat = (beat - np.mean(beat, axis=1)[:, None]) / np.std(beat, axis=1)[:, None]
        if nodata:
            beat_all = beat
            nodata = 0
        else:
            beat_all = np.concatenate([beat_all, beat], axis=0)
        beat = np.expand_dims(data[:,-args.ecglen:],axis=-1)
        beat = (beat-np.mean(beat,axis=1)[:,None])/np.std(beat,axis=1)[:,None]
        beat_all = np.concatenate([beat_all, beat], axis=0)
    return beat_all
# @tf.function
def load_raw_data_cpsc(datapath):
    ecgpaths = glob.glob(datapath + "*.mat")
    beat_all=[]
    nodata=1
    i=0
    for ecgpath in ecgpaths:
        print(i)
        i=i+1
        data=sio.loadmat(ecgpath)['ECG'][0][0][2]
        if len(data.reshape(-1))<args.ecglen:
            continue
        # data=StandardScaler().fit_transform(data.transpose()).transpose()
        data=np.expand_dims(data,axis=-1)
        Sig_Len=int((data.shape[1] // args.ecglen)*args.ecglen)
        rdata = data[:, :Sig_Len]
        beat = rdata.reshape(-1, args.ecglen, 1, order='C')
        if len(beat)==0:
            continue
        beat = (beat-np.mean(beat,axis=1)[:,None])/np.std(beat,axis=1)[:,None]
        if nodata:
            beat_all = beat
            nodata = 0
        else:
            beat_all = tf.concat([beat_all, beat], axis=0)
        beat = data[:,-args.ecglen:,:]
        beat = (beat-np.mean(beat,axis=1)[:,None])/np.std(beat,axis=1)[:,None]
        beat_all = tf.concat([beat_all, beat], axis=0)
    beat_all=np.asarray(beat_all)
    return beat_all

def datatorecord(tfrecordwriter,ecgs):
    writer = tf.io.TFRecordWriter(tfrecordwriter)  # 1. 定义 writer对象，创建tfrecord文件，输入的为文件名
    for i in range(ecgs.shape[0]):
        ecg=ecgs[i]
        ecg=np.asarray(ecg).astype(np.float32).tobytes()
        """ 2. 定义features """
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'ecg': tf.train.Feature(bytes_list = tf.train.BytesList(value=[ecg]))
                }))
        """ 3. 序列化,写入"""
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def decode_tfrecords4c(example):
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description = {
        'ecg': tf.io.FixedLenFeature([], tf.string)
    }
    # 按照feature_description解码
    feature_dict = tf.io.parse_single_example(example, feature_description)
    # 由bytes码转化为tf.float32
    ecg = (tf.io.decode_raw(feature_dict['ecg'], out_type=tf.float32))
    ecg=tf.reshape(ecg,[args.ecglen,1])
    return ecg

def read_tfrecords(tfrecord_file):
    #读取文件,数据预处理的一部分
    dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    dataset = dataset.map(decode_tfrecords4c,num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 解析数据
    return dataset

if __name__=='__main__':
    path = './single_arm/'
    pklpath0 = path+'SA_noised_data.pkl'
    pklpath1 = path+'CP_noised_data.pkl'
    pklpath  = path + 'noised_data.pkl'
    # print(os.path.exists(pklpath))
    if os.path.exists(pklpath):
        if os.path.exists(pklpath0):
            with open(pklpath0, 'rb') as file:  # 用with的优点是可以不用写关闭文件操作
                beat_SA = pickle.load(file)
        else:
            beat_SA = load_raw_data(datapath='./Device_Data2/')
            file = open(pklpath0, 'wb')
            pickle.dump(beat_SA, file)
            file.close()
        beat_CP=load_raw_data_cpsc('./CPSC2018/')
        beat_CP = np.delete(beat_CP, np.where(np.isnan(beat_CP) == 1), axis=0)

        file = open(pklpath1, 'wb')
        pickle.dump(beat_CP, file)
        file.close()

        beat_all = np.concatenate([beat_SA, beat_CP], axis=0)



        file = open(pklpath, 'wb')
        pickle.dump((beat_all), file)
        file.close()

        # 制作成tf数据集
        CPpath= './CPset'
        datatorecord(CPpath,beat_CP)

        SApath= './SAset'
        datatorecord(SApath,beat_SA)

        allpath= './trainset'
        datatorecord(allpath,beat_all)


