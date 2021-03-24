import tensorflow as tf
import numpy as np

def img_to_tensor(path):
    """
    Reads an image from a given path and preprocesses it.
    :param path: (string) path to specific image
    :return:
    """

    # Load image as tensor
    img = tf.io.read_file(path)

    # change format to float32-tensor with 3 color channels
    img = tf.image.decode_image(img, 3)
    img = tf.image.convert_image_dtype(img, tf.float32) # convert to [0-1]

    # normalize the shape (longer side gets 512 pixels)
    shape_original = tf.cast(tf.shape(img)[:-1], tf.float32)
    shape_normalized = tf.cast(shape_original * 512 / max(shape_original), tf.int32)
    img = tf.image.resize(img, shape_normalized)
    img = img[tf.newaxis, :]

    return img


def tensor_to_img(tensor_img):
    """
    Creates an image from a given tensor-image.
    :param tensor_img: (tensor) tensor image to convert
    :return: (array) image in the range of [0,255]
    """

    img = np.array(tensor_img[0,:,:,:] * 255, dtype=np.uint8)  # transfer back to range[0,255]

    return img
