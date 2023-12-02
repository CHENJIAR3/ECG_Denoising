# 主函数
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import numpy as np
import matplotlib.pyplot as plt
# 把单导联心电图作为输入信号
from model_structure import *
import pickle
import copy
import tensorflow as tf
from  utils import args
import warnings
from get_ecgdata import read_tfrecords
# 忽略所有警告
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
strategy = tf.distribute.MirroredStrategy()
def extract(a, t, x_shape):
    batch_size, sequence_length, _ = a.shape
    t_shape = tf.shape(t)
    out = tf.gather(a, t, axis=-1)
    out = tf.reshape(out, (batch_size, t_shape[0], *((1,) * (len(x_shape) - 1))))
    return out

class Trainer:
    def __init__(self, epochs, lr, time_steps, ecglen=256):
        self.ecglen = ecglen
        self.time_steps = time_steps
        self.epochs = epochs
        self.lr = lr
        with strategy.scope():

            self.model = unet3plus()
            self.model0 = tf.keras.models.load_model(bestpath)
            self.model.set_weights(self.model0.get_weights())
            self.forward_noiser = ForwardDiffusion(self.time_steps)
            self.alphas = self.forward_noiser.alphas
            self.betas = self.forward_noiser.betas
            self.alpha_hats = self.forward_noiser.alpha_hat

            self.mse = tf.keras.losses.MeanSquaredError()
            self.mae = tf.keras.losses.MeanAbsoluteError()
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.lr,
                decay_steps=100,
                decay_rate=0.9,
                staircase=True)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            self.loss_tracker = tf.keras.metrics.Mean()

            self.best_loss = np.inf
            self.patience=args.patience
            self.waiting=0
    def sample_time_step(self, size):
        return tf.experimental.numpy.random.randint(1, self.time_steps, size=(size,))

    @tf.function
    def train_step(self, ecgdata):
        def step_fn(ecgdata):
            t = self.sample_time_step(size=tf.shape(ecgdata)[0])
            mixed_data, noise = self.forward_noiser([ecgdata, t])
            t = tf.expand_dims(t, axis=-1)
            with tf.GradientTape() as tape:
                output = self.model([mixed_data,tf.cast(t,dtype=tf.int64)])
                loss = tf.reduce_mean((output-noise)*(output-noise),axis=0)
                    # self.mse(output, noise)
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            # 确保所有设备上的模型权重一致
            return tf.reduce_sum(loss)

        per_replica_losses = strategy.run(step_fn, args=(ecgdata,))

        return per_replica_losses


    @tf.function
    def test_step(self, ecgdata):
        # def step_test(ecgdata):
        t = self.sample_time_step(size=tf.shape(ecgdata)[0])
        mixed_data, noise= self.forward_noiser([ecgdata, t])
        t = tf.expand_dims(t, axis=-1)
        output = self.model([mixed_data,t])
        loss = tf.reduce_mean((output - noise)*(output - noise), axis=0)
        return tf.reduce_sum(loss)



    @tf.function(input_signature=[tf.TensorSpec(shape=(None, args.ecglen, 1), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None,1), dtype=tf.int32),
                                  tf.TensorSpec(shape=(None, args.ecglen, 1), dtype=tf.float32)])
    def reverse_diffusion(self, x_t, time,predicted_noise):

        alpha = tf.gather(self.alphas, time)[:, None]
        alpha_hat = tf.gather(self.alpha_hats, time)[:, None]
        alpha_hat_t_1 = tf.gather(self.alpha_hats, time - 1)[:, None]
        if time[0] > 1:
            noise = tf.cast((tf.random.normal(shape=tf.shape(x_t))), dtype=tf.float32)
        else:
            noise = tf.zeros_like(x_t)

        x_t_1 = (1 / tf.sqrt(alpha)) * (x_t - ((1 - alpha) / (tf.sqrt(1 - alpha_hat))) * predicted_noise) +tf.sqrt((1-alpha)*(1-alpha_hat_t_1)/(1-alpha_hat))*noise
        return  x_t_1

    def sample(self,ecgdata,epoch):
        no_of=1
        rect = np.array([0.1, 0.7, 0.8, 0.2])
        for i in range(no_of):
            plt.axes(rect)
            plt.gca().set_yticklabels([])
            plt.gca().set_xticklabels([])
            plt.plot(ecgdata[i, :, 0], 'b')
            plt.title(" Input " + str(i) + " in epoch " + str(epoch))
            plt.grid(True)

        # xt=tf.random.normal(shape=tf.shape(ecgdata))
        # xt=ecgdata
        xt = tf.cast(ecgdata, dtype=tf.float32)

        for t in reversed(range(1, self.time_steps)):
            time = tf.repeat(tf.constant(t,dtype=tf.int32), repeats=xt.shape[0], axis=0)
            time = tf.expand_dims(time, axis=-1)
            predicted_noise=self.model([xt,time],training=False)
            xt = self.reverse_diffusion(x_t=xt,time=time,predicted_noise=predicted_noise)

        rect[1]-=0.3
        for i in range(no_of):
            plt.axes(rect)
            plt.gca().set_yticklabels([])
            plt.gca().set_xticklabels([])
            plt.plot(xt[i, :, 0], 'k')
            plt.title("Output " + str(i) + " in epoch " + str(epoch))
            plt.grid(True)

        xt=tf.cast(tf.random.normal(shape=tf.shape(ecgdata)), dtype=tf.float32)
        # xt = ecgdata
        for t in reversed(range(1, self.time_steps)):
            time = tf.repeat(tf.constant(t, dtype=tf.int32), repeats=xt.shape[0], axis=0)
            time = tf.expand_dims(time, axis=-1)
            predicted_noise = self.model([xt, time],training=False)
            xt = self.reverse_diffusion(x_t=xt, time=time, predicted_noise=predicted_noise)
            # xt = (xt - tf.reduce_mean(xt, axis=1,keepdims=True)) / (tf.math.reduce_std(xt, axis=1,keepdims=True))
            # xt = tf.clip_by_value(xt, -3, 3)
        print(tf.reduce_max(xt))
        rect[1] -= 0.3
        for i in range(no_of):
            plt.axes(rect)
            plt.gca().set_yticklabels([])
            plt.gca().set_xticklabels([])
            # plt.plot(ecgdata[i,:,0],'b')
            plt.plot(xt[i, :, 0], 'k')
            plt.title("Noise Diffusion: " + str(i) + " in epoch " + str(epoch))
            plt.grid(True)
        plt.show()
    def train(self, train_data):
        for epoch in range(self.epochs):

            self.loss_tracker.reset_states()
            print("Epoch :", epoch)
            progbar = tf.keras.utils.Progbar(target=None, stateful_metrics=['loss', 'val_loss'])
            train_loss_c=0
            for train_batch, ecgdata in enumerate(train_data):

                train_loss = self.train_step(ecgdata)
                non_empty_losses = [loss for loss in train_loss.values if np.isnan(loss) == False]
                train_loss = tf.reduce_mean(non_empty_losses)
                train_loss_c+=train_loss.numpy()
                progbar.update(train_batch, values=[('loss', train_loss)],
                           finalize=False)
            tf.debugging.check_numerics(train_loss, "Loss contains NaN or Inf")

            if epoch%20==0:
                # ecgdata=valds.take(1)
                # ecgdata=next(iter(ecgdata))
                ecgdata = next(iter(trainds)).values[0]
                #ecgdata = ecgdata.values[0]
                self.sample(ecgdata,epoch)

            progbar.update(train_batch, values=[
                ('loss', train_loss),
            ], finalize=True)
            # val_loss_c=0
            # for val_batch,ecgdata in enumerate(valds):
            #     val_loss=self.test_step(ecgdata)
            #     # non_empty_losses = [loss for loss in val_loss.values if np.isnan(loss) == False]
            #     val_loss_c+= tf.reduce_mean(val_loss)
            progbar.update(train_batch, values=[('loss', train_loss_c)],
                           finalize=True)


            if train_loss_c < self.best_loss:
                self.best_loss = train_loss_c
                tf.keras.models.save_model(self.model,bestpath)
                print("Best Weights saved...")
                self.waiting=0
            else:
                self.waiting+=1
            if self.waiting>self.patience:
                print("Finished in advance")
                break






# Create a 'Trainer' instance and call 'instance.train(train,val)' to train
if __name__ == "__main__":
    path = './single_arm/'
    pklpath=path+'CP_noised_data.pkl'

    targetpath=path+'SA_noised_data.pkl'

    # with open(pklpath, 'rb') as file:  # 用with的优点是可以不用写关闭文件操作
    #     x_train = pickle.load(file)
    with open(targetpath, 'rb') as file:  # 用with的优点是可以不用写关闭文件操作
        x_val = pickle.load(file)
    # trainds=tf.data.Dataset.from_tensor_slices((x_train)).batch(args.bs).shuffle(10000)
    valds=tf.data.Dataset.from_tensor_slices((x_val)).batch(args.bs).shuffle(1024)

    trainpath= './CPset'

    options = tf.data.Options()

    # Set the auto-shard policy to AutoShardPolicy.DATA
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    trainds = read_tfrecords(trainpath).batch(args.bs).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).shuffle(1024)
    trainds=trainds.with_options(options)
    trainds=strategy.experimental_distribute_dataset(trainds)
    bestpath = "./Denoiser"

    trainer_df = Trainer(epochs=args.epochs, lr=args.lr, time_steps=args.time_steps, ecglen=args.ecglen)
    trainer_df.train(trainds)