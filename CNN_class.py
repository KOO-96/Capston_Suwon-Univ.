import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical


class CNN:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
        maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flatten = Flatten()(maxpool2)
        fc1 = Dense(128, activation='relu')(flatten)
        output_layer = Dense(self.num_classes, activation='softmax')(fc1)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        return x_train, y_train, x_test, y_test

    def train(self, x_train, y_train, x_test, y_test, epochs=20, batch_size=128):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
        return history

    def evaluate(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        return test_loss, test_acc