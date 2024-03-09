import tensorflow as tf
from keras.models import load_model
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import keras.utils as image
from keras.utils import load_img
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

new_model = tf.keras.models.load_model(r"C:\dogImages\model_fit_24-0.82.h5")
input = r"C:\Users\sebas\OneDrive\Desktop\New folder\lab.jpeg"
def load_image(filename):
    IMAGE_SIZE = IMAGE_SIZE = [224, 224]
    img = cv2.imread(filename)
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]) )
    img = img /255
    
    return img
data=tf.keras.utils.image_dataset_from_directory(r"C:\dogImages\train")
classes=data.class_names


def predict(image):
    probabilities = new_model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}

img = load_image(str(input))
prediction = predict(img)
print("ACTUAL CLASS: %s, PREDICTED: class: %s, confidence: %f" % (os.path.basename(input), list(prediction.keys())[0], list(prediction.values())[0]))
plt.imshow(img)
plt.show()