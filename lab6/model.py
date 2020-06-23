from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import tensorflow.keras.backend as K


class Autoencoder_models:
    def __init__(self, height, width, channels):
        self.height = height
        self.width = width
        self.channels = channels
        # Размерность кодированного представления
        self.encoding_dim = 85

    def create_dense_ae(self):
        # Энкодер
        input_img = Input(shape=(self.height, self.width, self.channels))
        flat_img = Flatten()(input_img)
        encoded = Dense(self.encoding_dim, activation='relu')(flat_img)

        # Декодер
        input_encoded = Input(shape=(self.encoding_dim,))
        flat_decoded = Dense(self.height * self.width * self.channels, activation='sigmoid')(input_encoded)
        decoded = Reshape((self.height, self.width, self.channels))(flat_decoded)

        encoder = Model(input_img, encoded, name="encoder")
        decoder = Model(input_encoded, decoded, name="decoder")

        autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
        return autoencoder

    def create_deep_conv_ae(self):
        input_img = Input(shape=(self.height, self.width, self.channels))

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        return autoencoder

    def create_denoising_model(self, autoencoder, batch_size):
        def add_noise(x):
            noise_factor = 0.5
            x = x + K.random_normal(x.get_shape(), 0.5, noise_factor)
            x = K.clip(x, 0., 1.)
            return x

        input_img  = Input(batch_shape=(batch_size, self.height, self.width, self.channels))
        noised_img = Lambda(add_noise)(input_img)

        noiser = Model(input_img, noised_img, name="noiser")
        denoiser_model = Model(input_img, autoencoder(noiser(input_img)), name="denoiser")
        return denoiser_model
