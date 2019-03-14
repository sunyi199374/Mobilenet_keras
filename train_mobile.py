from __future__ import print_function
import keras
import mobilenet_keras
import data_pre
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from mobilenet_keras import MobileNet

total_samples = 1514
batch_size = 8
#num_classes = 5

learning_rate = 0.0001
spepochs = total_samples/batch_size
print(spepochs)
epochs = 100

# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = data_pre.load_image('/home/e/mycnn_data/data_filter')
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='/home/e/mycnn_data/data_no_filter',
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory='/home/e/mycnn_data/data_no_filter',
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

MODEL = MobileNet()


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=learning_rate, decay=1e-6)

# Let's train the model using RMSprop
MODEL.model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

mc = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', verbose = 1, period=5, save_best_only=True)
es = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
MODEL.model.fit_generator(
        train_generator,
        steps_per_epoch=spepochs,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=800,
        callbacks=[mc,es]
        )

#score = MODEL.model.evaluate(x_test, y_test, verbose=0)

score = MODEL.model.evaluate_generator(test_generator, 400)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
MODEL.model.save('mobilenet_parts_model.h5')