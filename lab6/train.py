import matplotlib
import pytorch_lightning as pl
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers
import tensorflow as tf

matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
import cv2
from model import Autoencoder_models

TRAIN_DATA = 'data/train'
MODEL = 'output/model.h5'
PLOT_PATH = 'plot.png'

EPOCHS = 10
BATCH_SIZE = 3

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# инициализируем данные и метки
print("[INFO] loading images...")
data = []
image_paths = list(paths.list_images(TRAIN_DATA))

# цикл по изображениям
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    data.append(image)

print('[INFO] images loaded')

# разбиваем данные на обучающую (80%) и тестовую выборки (20%)
trainX, testX = train_test_split(data, test_size=0.2)

trainX = np.asarray(trainX).astype("float32") / 255.0
testX = np.asarray(testX).astype("float32") / 255.0

print("[INFO] building autoencoder...")

autoencoder_models = Autoencoder_models(IMAGE_HEIGHT, IMAGE_WIDTH, 3)

autoencoder_deep = autoencoder_models.create_deep_conv_ae()
autoencoder_deep.summary()

denoiser_model = autoencoder_models.create_denoising_model(autoencoder_deep, BATCH_SIZE)
denoiser_model.summary()
denoiser_model.compile(optimizer='adam', loss='binary_crossentropy')

H = denoiser_model.fit(trainX, trainX,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   validation_data=(testX, testX))

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)

denoiser_model.save(MODEL)
