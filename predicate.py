import os.path
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
from keras.preprocessing import image

from train import train
from train import create_model


# Формат изображений.
img_width, img_height = [150] * 2

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'


def recognize(target):
    model = create_model()


    #loading weights
    if os.path.exists('weights.h5'):
        model.load_weights('weights.h5')
    else:
        train(model)

    if os.path.isfile(target):
        img = image.load_img(target, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        raw_img = matplotlib.image.imread(target)
        plt.imshow(raw_img)
        if prediction:
            result = 'butterflies'
            plt.text(0, -20, 'Это изображение бабочки.', fontsize=20)
        else:
            result = 'flowers'
            plt.text(150, -50, 'Это изображение цветов.', fontsize=20)
        plt.axis("off")
        plt.show()
    else:
        raise IOError('No such file')