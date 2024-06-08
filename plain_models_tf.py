import tensorflow as tf
import torch
import numpy as np

from tools import load_tf_data, load_data, load_torch_data
from plain_models import CryptoNet_Digits_helayers, CryptoNet_MNIST_helayers, test


class SquareActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SquareActivation, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.square(inputs)


class PolyActivation(tf.keras.layers.Layer):
    def __init__(self, coefs):
        super(PolyActivation, self).__init__()
        self.coefs = coefs
    
    def call(self, inputs):
        return tf.math.polyval(self.coefs, inputs)
    
    def get_config(self):
        config = super().get_config()
        config["coefs"] = self.coefs
        return config


def CryptoNet_MNIST_tf():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=5, kernel_size=5, strides=3, padding='valid', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(SquareActivation())
    model.add(tf.keras.layers.Dense(100))
    model.add(SquareActivation())
    model.add(tf.keras.layers.Dense(10))

    return model


def CryptoNet_DIGITS_tf():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, strides=1, padding='valid', input_shape=(8, 8, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(tf.keras.layers.Dense(10))

    return model


def CryptoNet_DIGITS_tf_poly():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, strides=1, padding='valid', input_shape=(8, 8, 1)))
    model.add(tf.keras.layers.Flatten())
    # sigmoid activation: x = 0.5 + 0.197 * x - 0.004 * (x ** 3)  # sigmoid
    model.add(PolyActivation([-0.004, 0., 0.197, 0.5]))
    model.add(tf.keras.layers.Dense(64))
    model.add(PolyActivation([-0.004, 0., 0.197, 0.5]))
    model.add(tf.keras.layers.Dense(10))

    return model


def tf_train(data_name):
    if data_name == "mnist":
        model = CryptoNet_MNIST_tf()
        x_train, x_test, y_train, y_test = load_tf_data(data_name)
    elif data_name == "digits":
        model = CryptoNet_DIGITS_tf()
        x_train, x_test, y_train, y_test = load_tf_data(data_name)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=64,
            epochs=100,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_json = model.to_json()
    with open(f'./pretrained/{data_name}_plain_tf.json', "w") as fp:
        fp.write(model_json)
        
    model.save_weights(f'./pretrained/{data_name}_plain_tf.h5')
    model.save(f'./pretrained/{data_name}_plain_tf_full.h5')


def digits_ploy_convert():
    model = CryptoNet_DIGITS_tf_poly()
    x_train, x_test, y_train, y_test = load_tf_data("digits")

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model.load_weights('./pretrained/digits_plain_tf.h5')
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                metrics=['accuracy'])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_json = model.to_json()
    with open(f'./pretrained/digits_plain_tf.json', "w") as fp:
        fp.write(model_json)
        
    model.save_weights(f'./pretrained/digits_plain_tf.h5')


def torch_convert(data_name):
    if data_name == "digits":
        torch_model = CryptoNet_Digits_helayers()
        train_loader, test_loader = load_data("digits")
        tf_model = tf.keras.models.load_model(f'./pretrained/{data_name}_plain_tf_full.h5')
        tf_weights = tf_model.get_weights()
    elif data_name == "mnist":
        torch_model = CryptoNet_MNIST_helayers()
        train_loader, test_loader = load_torch_data("mnist")
        tf_model = CryptoNet_MNIST_tf()
        tf_model.load_weights(f'./pretrained/{data_name}_plain_tf.h5')
        tf_weights = tf_model.get_weights()

    torch_model.eval()

    with torch.no_grad():
        torch_weights = torch_model.state_dict()
        torch_weights['conv1.weight'] = torch.from_numpy(np.transpose(tf_weights[0], (3, 2, 0, 1)))
        torch_weights['conv1.bias'] = torch.from_numpy(tf_weights[1])

        torch_weights['fc1.weight'] = torch.from_numpy(np.transpose(tf_weights[2], (1, 0)))
        torch_weights['fc1.bias'] = torch.from_numpy(tf_weights[3])
        
        torch_weights['fc2.weight'] = torch.from_numpy(np.transpose(tf_weights[4], (1, 0)))
        torch_weights['fc2.bias'] = torch.from_numpy(tf_weights[5])

        torch_model.load_state_dict(torch_weights)

    acc = test(torch_model, test_loader)
    print(f"digits: {acc:.2f}%")

    torch.save(torch_model.state_dict(), f'./pretrained/{data_name}_plain_tf.pt')

if __name__ == "__main__":
    tf_train("digits")
    tf_train("mnist")

    digits_ploy_convert()

    torch_convert("digits")
    torch_convert("mnist")
