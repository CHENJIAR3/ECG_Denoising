# 模型架构
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class forward_diffusion_helper:
    def __init__(self, time_steps, beta_start=1e-4, beta_end=1e-1):
        self.time_steps = tf.cast(time_steps, tf.float32)
        self.s = 1.0
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = self.s*self.compute_betas()

        self.alphas = 1. - self.betas
        self.alpha_hat = tf.math.cumprod(self.alphas)

    def compute_betas(self,schedule='quad'):
        if schedule == 'linear':
            betas = tf.linspace(self.beta_start, self.beta_end, int(self.time_steps))
        elif schedule == "quad":
            betas = tf.linspace(tf.sqrt(self.beta_start), tf.sqrt( self.beta_end), int(self.time_steps)) ** 2
        elif schedule == 'square':
            betas = tf.sqrt(tf.linspace((self.beta_start)**2, ( self.beta_end)**2, int(self.time_steps)))
        elif schedule == "sigmoid":
            betas = tf.linspace(-6.0, 6.0, int(self.time_steps))
            betas = tf.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        return betas

# 下述是前向扩散过程
class ForwardDiffusion:
    def __init__(self, time_steps):
        self.time_steps = time_steps
        # 所有的参数，α、β都在params里面
        self.params = forward_diffusion_helper(self.time_steps)
        self.alpha_hat = self.params.alpha_hat
        self.alphas = self.params.alphas
        self.betas = self.params.betas

    def __call__(self, inputs):
        # 下面开始call这个函数
        x, t = inputs
        x = tf.cast(x,dtype=tf.float32)
        noise = tf.cast(tf.random.normal(shape=tf.shape(x)),dtype=tf.float32)
        # 改动
        # 变成3维的尺度
        sqrt_alpha_hat = tf.math.sqrt(
            tf.gather(self.alpha_hat, t)
        )[:, None,None]
        sqrt_one_minus_alpha_hat = tf.math.sqrt(
            1. - tf.gather(self.alpha_hat, t)
        )[:, None,None]

        # 开始加噪
        noised_x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return noised_x, noise

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed):
        super(PositionalEmbedding, self).__init__()
        self.embed = embed
        #build是接受一个输入尺寸
        self.emb = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed),
            tf.keras.layers.Activation("swish"),
            tf.keras.layers.Dense(self.embed)
        ])
        self.emb.build((None, self.embed))

    def call(self, t):
        # embed/2
        embed = self.embed / 2
        rates = 1.0 / (10000 ** (tf.range(0, embed, dtype=tf.float32) / embed))
        rates = tf.expand_dims(rates, axis=0)
        # 时间信息
        t = tf.cast(t, tf.float32)
        sines = tf.sin(t * rates)
        cosines = tf.cos(t * rates)
        embeddings = tf.concat([sines, cosines], axis=-1)
        return self.emb(embeddings)

    
def conv_block(x,t, kernels, kernel_size=(3), strides=(1), padding='same',is_norm=True, is_activation=True, n=2):

    for i in range(n):
        x = tf.keras.layers.Conv1D(filters=kernels, kernel_size=kernel_size,
                            padding=padding, strides=strides,
                            kernel_initializer=tf.keras.initializers.he_normal(seed=5))(x)

        if is_activation:
            x = tfa.layers.GELU()(x)
        if is_norm:
            x = tfa.layers.InstanceNormalization()(x)
        # if is_activation:
            # x = tfa.layers.GELU()(x)
    x_emb=tf.keras.layers.Dense(kernels,activation='elu')(t)
    x_emb=tf.keras.layers.Reshape((1,kernels))(x_emb)
    x = x + x_emb
    if is_activation:
        x = tfa.layers.GELU()(x)
    if is_norm:
        x = tfa.layers.InstanceNormalization()(x)

    return x

def unet3plus(input_size=(None,1), output_channels=1):
    """ UNet3+ base model """
    filters = [16,64,128,256,1024]

    input_layer = tf.keras.layers.Input(
        shape=input_size,
        name="input_layer"
    )
    time_layer = tf.keras.layers.Input(
        shape=(1),
        name="time_embed"
    )
    time_embed=PositionalEmbedding(32)(time_layer)

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer,time_embed, filters[0])

    # block 2
    e2 = tf.keras.layers.MaxPool1D(pool_size=(4))(e1)
    e2 = conv_block(e2,time_embed, filters[1])

    # block 3
    e3 = tf.keras.layers.MaxPool1D(pool_size=(4))(e2)
    e3 = conv_block(e3,time_embed, filters[2])

    # block 4
    e4 = tf.keras.layers.MaxPool1D(pool_size=(4))(e3)  #
    e4 = conv_block(e4,time_embed, filters[3])  #

    # block 5
    # bottleneck layer
    e5 = tf.keras.layers.MaxPool1D(pool_size=(4))(e4)
    e5 = conv_block(e5,time_embed, filters[4])  #
    
    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = tf.keras.layers.MaxPool1D(pool_size=(64))(e1)
    e1_d4 = conv_block(e1_d4,time_embed, cat_channels, n=1)

    e2_d4 = tf.keras.layers.MaxPool1D(pool_size=(16))(e2)
    e2_d4 = conv_block(e2_d4,time_embed, cat_channels, n=1)

    e3_d4 = tf.keras.layers.MaxPool1D(pool_size=(4))(e3)
    e3_d4 = conv_block(e3_d4,time_embed, cat_channels, n=1)

    e4_d4 = conv_block(e4,time_embed, cat_channels, n=1)

    e5_d4 = tf.keras.layers.UpSampling1D(size=(4))(e5)
    e5_d4 = conv_block(e5_d4,time_embed, cat_channels, n=1)

    d4 = tf.keras.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4,time_embed, upsample_channels, n=1)

    """ d3 """
    e1_d3 = tf.keras.layers.MaxPool1D(pool_size=(16))(e1)
    e1_d3 = conv_block(e1_d3,time_embed, cat_channels, n=1)

    e2_d3 = tf.keras.layers.MaxPool1D(pool_size=(4))(e2)
    e2_d3 = conv_block(e2_d3,time_embed, cat_channels, n=1)

    e3_d3 = conv_block(e3,time_embed, cat_channels, n=1)

    e4_d3 = tf.keras.layers.UpSampling1D(size=(4))(d4)
    e4_d3 = conv_block(e4_d3,time_embed, cat_channels, n=1)
    e5_d3 = tf.keras.layers.UpSampling1D(size=(16))(e5)
    e5_d3 = conv_block(e5_d3,time_embed, cat_channels, n=1)

    d3 = tf.keras.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
    d3 = conv_block(d3,time_embed, upsample_channels, n=1)

    """ d2 """
    e1_d2 = tf.keras.layers.MaxPool1D(pool_size=(4))(e1)
    e1_d2 = conv_block(e1_d2,time_embed, cat_channels, n=1)

    e2_d2 = conv_block(e2,time_embed, cat_channels, n=1)

    d3_d2 = tf.keras.layers.UpSampling1D(size=(4))(d3)
    d3_d2 = conv_block(d3_d2,time_embed, cat_channels, n=1)

    d4_d2 = tf.keras.layers.UpSampling1D(size=(16))(d4)
    d4_d2 = conv_block(d4_d2,time_embed, cat_channels, n=1)

    e5_d2 = tf.keras.layers.UpSampling1D(size=(64))(e5)
    e5_d2 = conv_block(e5_d2,time_embed, cat_channels, n=1)

    d2 = tf.keras.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2,time_embed, upsample_channels, n=1)

    """ d1 """
    e1_d1 = conv_block(e1,time_embed, cat_channels, n=1)

    d2_d1 = tf.keras.layers.UpSampling1D(size=(4))(d2)
    d2_d1 = conv_block(d2_d1,time_embed, cat_channels, n=1)

    d3_d1 = tf.keras.layers.UpSampling1D(size=(16))(d3)
    d3_d1 = conv_block(d3_d1,time_embed, cat_channels, n=1)

    d4_d1 = tf.keras.layers.UpSampling1D(size=(64))(d4)
    d4_d1 = conv_block(d4_d1,time_embed, cat_channels, n=1)

    e5_d1 = tf.keras.layers.UpSampling1D(size=(256))(e5)
    e5_d1 = conv_block(e5_d1,time_embed, cat_channels, n=1)

    d1 = tf.keras.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = conv_block(d1,time_embed, upsample_channels, n=1)

    output = conv_block(d1,time_embed, output_channels, n=1)

    return tf.keras.Model(inputs=[input_layer,time_layer], outputs=[output], name='Models_Unet3')

if __name__=='__main__':
    model=unet3plus()
    model.summary()