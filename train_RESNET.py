import os
from data_utils_RESNET import train_generator, val_batch_generator
from model_utils_RESNET import generate_RESNET_model
from train_utils_RESNET import TensorBoardBatch
from keras.callbacks import ModelCheckpoint


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
nb_train_images = 512 # con cuantas imagenes se quiere entrenar
batch_size = 32 #se toman de poco
epocas=256 #cuantas veces se pasa cada batch por la red
model = generate_RESNET_model(lr=1e-3)
model.summary()

# continue training if weights are available
#if os.path.exists('weights/mobilenet_model.h5'):
#    model.load_weights('weights/mobilenet_model.h5')

# use Batchwise TensorBoard callback
tensorboard = TensorBoardBatch(batch_size=batch_size)
checkpoint = ModelCheckpoint('weights/pesos_resnet__total_loss__lr_1e-3.h5', monitor='loss', verbose=1,
                             save_best_only=True, save_weights_only=True)
callbacks = [checkpoint, tensorboard]

#model.fit_generator(image_a_b_gen(batch_size), epochs=epocas, steps_per_epoch=5)

gen=train_generator(batch_size)
val=val_batch_generator(batch_size)
model.fit_generator(gen,
                    steps_per_epoch=nb_train_images // batch_size,
                    epochs=epocas,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val,
                    validation_steps=1
                    )
