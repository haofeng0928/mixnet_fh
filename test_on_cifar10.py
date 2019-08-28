from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Dropout, Dense, Activation, MaxPooling2D
from keras.layers import BatchNormalization, ReLU, GlobalAveragePooling2D
from keras.models import Model

from custom_objects import MixNetConvInitializer, MixNetDenseInitializer
from mixnets import GroupedConv2D, MixNetBlock

import os

from mixnets import MixNetSmall

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

batch_size = 32
num_classes = 10
epochs = 20
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape=(32, 32, 3))

# Stem part
x = inputs
x = GroupedConv2D(
    filters=16,
    kernel_size=[3],
    strides=[2, 2],
    kernel_initializer=MixNetConvInitializer(),
    padding='same',
    use_bias=False)(x)
x = BatchNormalization(
    axis=-1,
    momentum=0.99,
    epsilon=1e-3)(x)
x = ReLU()(x)

# Blocks part
x = Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:])(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

# 替换部分
# x = Conv2D(64, (3, 3), padding='same', activation='relu', strides=(2, 2))(x)
# # x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)
# x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
# x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
# # x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)
# x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
# x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
# # x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)

x = MixNetBlock(input_filters=64,
                output_filters=256,
                dw_kernel_size=[3, 5, 7],
                expand_kernel_size=[1],
                project_kernel_size=[1],
                strides=[2, 2],
                expand_ratio=6,
                se_ratio=0.5,
                id_skip=True,
                drop_connect_rate=0,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                swish=True,
                data_format='channels_last')(x)

# Head part
x = GroupedConv2D(
    filters=512,  # 1536
    kernel_size=[1],
    strides=[1, 1],
    kernel_initializer=MixNetConvInitializer(),
    padding='same',
    use_bias=False)(x)
x = BatchNormalization(
    axis=-1,
    momentum=0.99,
    epsilon=1e-3)(x)
x = ReLU()(x)

x = GlobalAveragePooling2D(data_format='channels_last')(x)
x = Dropout(0.2)(x)
x = Dense(10, kernel_initializer=MixNetDenseInitializer())(x)
x = Activation('softmax')(x)
outputs = x
model = Model(inputs, outputs)

# model = MixNetSmall(input_shape=(32, 32, 3), include_top=True, weights=False, classes=10)
model.summary()

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4,
                        steps_per_epoch=x_train.shape[0])

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


