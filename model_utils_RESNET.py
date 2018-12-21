from keras.layers import Conv2D, Input, Reshape, RepeatVector, concatenate, UpSampling2D, Flatten, Conv2DTranspose
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from keras.losses import mean_squared_error
from keras.optimizers import Adam

mse_weight = 1.0 #1e-3

# set these to zeros to prevent learning
perceptual_weight = 1. / (2. * 256. * 256.) # scaling factor
attention_weight = 1.0 # 1.0


def loss_with_metrics(img_ab_out, img_ab_true, name=''):
    # Loss is mean square erros
    cost = tf.reduce_mean(
        tf.squared_difference(img_ab_out, img_ab_true), name="mse")
    # Metrics for tensorboard
    #summary = tf.summary.scalar('cost ' + name, cost)
    return cost

# shows the minimum value of the AB channels
def y_true_min(yt, yp):
    return K.min(yt)

 # shows the maximum value of the RGB AB channels
def y_true_max(yt, yp):
    return K.max(yt)

# conda install  tensorflow-gpu==1.12  cudatoolkit==9.0  cudnn==7.1.2  h5py


# shows the minimum value of the predicted AB channels
def y_pred_min(yt, yp):
    return K.min(yp)


# shows the maximum value of the predicted AB channels
def y_pred_max(yt, yp):
    return K.max(yp)


def gram_matrix(x):
    assert K.ndim(x) == 4

    with K.name_scope('gram_matrix'):
        if K.image_data_format() == "channels_first":
            batch, channels, width, height = K.int_shape(x)
            features = K.batch_flatten(x)
        else:
            batch, width, height, channels = K.int_shape(x)
            features = K.batch_flatten(K.permute_dimensions(x, (0, 3, 1, 2)))

        gram = K.dot(features, K.transpose(features)) # / (channels * width * height)
    return gram


def l2_norm(x):
    return K.sqrt(K.sum(K.square(x)))


def attention_vector(x):
    if K.image_data_format() == "channels_first":
        batch, channels, width, height = K.int_shape(x)
        filters = K.batch_flatten(K.permute_dimensions(x, (1, 0, 2, 3)))  # (channels, batch*width*height)
    else:
        batch, width, height, channels = K.int_shape(x)
        filters = K.batch_flatten(K.permute_dimensions(x, (3, 0, 1, 2)))  # (channels, batch*width*height)

    filters = K.mean(K.square(filters), axis=0)  # (batch*width*height,)
    filters = filters / l2_norm(filters)  # (batch*width*height,)
    return filters


def total_loss(y_true, y_pred):
    mse_loss = mse_weight * mean_squared_error(y_true, y_pred)
    perceptual_loss = perceptual_weight * K.sum(K.square(gram_matrix(y_true) - gram_matrix(y_pred)))
    attention_loss = attention_weight * l2_norm(attention_vector(y_true) - attention_vector(y_pred))

    return mse_loss + perceptual_loss + attention_loss


def generate_RESNET_model(lr=1e-3, img_size=256):

    embed_input = Input(shape=(1000,))
    # Encoder
    encoder_input = Input(shape=(img_size, img_size, 1,))
    encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)


    # Fusion: Donde entra la magia de Resnet
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

    # Decoder
    decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
    #decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)  # el ultimo tanh es porque lab se mueve entre -128 y 128: se necesita los negativos
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model.compile(optimizer=Adam(lr), loss=loss_with_metrics, metrics=[y_true_max,
                                                                y_true_min,
                                                                y_pred_max,
                                                                y_pred_min])

    print("Colorization model built and compiled")
    return model


if __name__ == '__main__':
    model = generate_RESNET_model()
    model.summary()

    #from keras.utils.vis_utils import plot_model

    #plot_model(model, to_file='skip_model.png', show_shapes=True)