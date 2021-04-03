import tensorflow as tf
import numpy as np
import cv2


def img2tensor(path):
    """
    Loads an image from a given path and preprocesses it for usage for the
    VGG19 network as described above.

    :param path: (string) path to an image
    return: (tensor) preprocessed image as tensor
    """

    # Load image as BGR, float32 image in the range [0,255]
    img = cv2.imread(path).astype(np.float32)

    # Save information about the original shape
    orig_shape = img.shape

    # Normalize (zero-center) the values according to imagenet mean of
    # respective color channel from imagenet dataset
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    # Convert image to tensor
    img_tensor = tf.convert_to_tensor(img)

    # Resize to 224 x 224 x 3 and add a new axis for batch size
    img_resized = tf.image.resize(img_tensor, (224,224))
    img_resized = img_resized[tf.newaxis, :]

    return img_resized, orig_shape


def tensor2img(tensor, orig_shape):
    """
    Inverse method for preprocessing function (img2tensor). This method
    takes an image created by the network and adapts the values such that it
    is an array in RGB colorspace, value range [0,255]. Also, the image is
    resized to its original shape.

    :param tensor: (tensor) input image as tensor
    :param orig_shape: (array) shape of original content image
    return: (array) image for plotting
    """

    # Convert tensor to array
    img = np.array(tensor)

    # Inverse operation to zero-centering in preprocessing function
    # (image in BGR color space)
    img[:, :, :, 0] += 103.939
    img[:, :, :, 1] += 116.779
    img[:, :, :, 2] += 123.68

    # All values must be in range [0,255], clip in case of rounding errors
    img_clipped = np.clip(img, 0., 255.)

    # Convert image to 3-channel RGB image
    img_rgb = img_clipped[0, :, :, ::-1].astype(np.uint8)

    # Resize such that size matches that of the original image
    img_resized = cv2.resize(img_rgb, (orig_shape[1], orig_shape[0]))

    return img_resized
