import tensorflow as tf
from skimage import io
import numpy as np
import os
import cv2

from preprocessing import img_to_tensor, tensor_to_img

CONTENT_TITLE_DICT = {}
STYLE_TITLE_DICT = {}
PATH = 'style-transfer/images'
PATH_STYLE = os.path.join(PATH, 'style')
PATH_CONTENT = os.path.join(PATH, 'content')
PATH_GENERATED = os.path.join(PATH, 'generated')


def make_dirs(paths):
    """
    Creates a new directory for given paths if directory does not exist yet.
    :param paths: (list) list of paths to create directories.
    :return: None
    """

    # Iterate over all given path strings and create directory if it does not exist.
    for path in paths:

        if not os.path.exists(path):
            os.makedirs(path)


def create_required_directories():
    """
    Creates the required dictionaries for this project.
    :return: None
    """

    # Define paths directories for images (subdivided into style, content and
    # generated images).

    paths = [PATH, PATH_STYLE, PATH_CONTENT, PATH_GENERATED]

    # Create directories if they do not exist.
    make_dirs(paths)


def update_dictionary(image_name, image_description, style_image):
    """
    Adds a new entry to either the content or the style dictionary.
    :param image_name: (key)
    :param image_description: (value)
    :return: None
    """

    if style_image:
        STYLE_TITLE_DICT[image_name] = image_description
    else:
        CONTENT_TITLE_DICT[image_name] = image_description


def test_jpg_or_png(string_to_test):
    """
    Tests whether a given string has the ending '.jpg' or '.png'.
    :param string_to_test:
    :return: (bool)
    """

    return string_to_test.endswith('.jpg') or string_to_test.endswith('.png')


def load_image_from_link(link, image_name, image_description, style_image=True):

    # Test if given image_name is valid
    assert test_jpg_or_png(image_name), 'Please provide the image name with the ending \'.jpg\' or \'.png\'.'
    assert test_jpg_or_png(link), 'Please provide a valid link. The given link is not a link to an image.'

    if style_image:
        image_type = 'style'
    else:
        image_type = 'content'

    # assert if path exists

    # image_path = os.path.join('/content', 'images', image_type, image_name)
    image_path = os.path.join('../images', image_type, image_name)  # For running notebook locally

    image = io.imread(link)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, image_rgb)

    update_dictionary(image_name, image_description, style_image)


def load_all_images(style_images, content_images):

    all_images = style_images + content_images

    for image in all_images:

        assert len(image) == 4, 'The provided image information is incorrect. It must be of type: link (string), image_name (string), image_description (string), style_image (bool)'

        link, image_name, image_description, style_bool = image

        # assert if first three are strings, forth is bool

        load_image_from_link(link, image_name, image_description, style_bool)


def test_preprocessing():

    path_style = os.path.join('../images', 'style')

    # Define path to test image
    img_path = os.path.join(path_style, 'the_scream.jpg')

    # Image to tensor
    img_tensor = img_to_tensor(img_path)
    assert tf.is_tensor(img_tensor), 'Image is not a tensor.'

    # Tensor to image
    img = tensor_to_img(img_tensor)

    assert isinstance(img, np.ndarray)
    # TODO: assertion, ob image vorher das gleiche ist wie nach processing
    # assert img_orig == img


def choose_images(content_image_name, style_image_name):
    """

    :param content_image_name:
    :param style_image_name:
    :return:
    """

    # CONTENT IMAGE
    image_content = img_to_tensor(os.path.join(PATH_CONTENT, content_image_name))

    # STYLE IMAGE (must be resized to content image size)
    image_style = img_to_tensor(os.path.join(PATH_STYLE, style_image_name))
    image_style = tf.image.resize(image_style, np.array(image_content).shape[1:3])

    return image_content, image_style


def clip_values(img):
    """
    Clip values of a given tensor-image to the range [0,1]
    :param img: (tensor) image
    :return: (tensor) image in range [0,1]
    """
    # TODO: Vielleicht eher in preprocessing?

    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)

