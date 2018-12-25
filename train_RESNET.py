import os
from data_utils_RESNET import train_generator, val_batch_generator
from model_utils_RESNET import generate_RESNET_model
from train_utils_RESNET import TensorBoardBatch
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
nb_train_images = 82000 # con cuantas imagenes se quiere entrenar
batch_size = 32 #se toman de poco
epocas=10000 #cuantas veces se pasa re recorre el dataset
model = generate_RESNET_model(lr=1e-4)
model.summary()



# use Batchwise TensorBoard callback
tensorboard = TensorBoardBatch(batch_size=batch_size)
checkpoint = ModelCheckpoint('weights/pesos_resnet_mse_l_1e-4.h5', monitor='loss', verbose=1,
                             save_best_only=True, save_weights_only=True)
early_stopping =EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=300,
                             verbose=0, mode='auto', baseline=None, restore_best_weights=False)
callbacks = [checkpoint, early_stopping,tensorboard]


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