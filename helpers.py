import tensorflow as tf
from skimage import io
import numpy as np
import os
import cv2

from processing import img2tensor, tensor2img

CONTENT_TITLE_DICT = {}
STYLE_TITLE_DICT = {}
PATH = 'images'
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
    Updates the dictionary that stores the image descriptions to use them
    as titles for visualization.

    :param image_name: (string) name of image file
    :param image_description: (string) description of image, e.g., its full title
    :param style_image: (bool) whether the given image is a style (True) or
        content (False) image
    :return: None
    """

    if style_image:
        STYLE_TITLE_DICT[image_name] = image_description
    else:
        CONTENT_TITLE_DICT[image_name] = image_description


def test_jpg_or_png(string_to_test):
    """
    Tests whether a given string has the ending '.jpg' or '.png'.
    :param string_to_test: (string) the string to test on
    :return: (bool) true if the string ended with one of the options
    """

    return string_to_test.endswith('.jpg') or string_to_test.endswith('.png')


def load_image_from_link(link, image_name, image_description, style_image=True):
    """
    Load an image from a given string and save it with a given file name in
    the style or content image directory. Also, the TITLE_DICT is updated.

    :param link: (string) link to the image to load
    :param image_name: (string) name of the image file
    :param image_description: (string) description of the image
    :param style_image: (bool) whether to load a style or content image - True for
        style image, False for content image
    return: None
    """

    # Test if given image_name is valid
    assert test_jpg_or_png(image_name), 'Please provide the image name with the ending \'.jpg\' or \'.png\'.'
    assert test_jpg_or_png(link), 'Please provide a valid link. The given link is not a link to an image.'

    # Store the image in the correct directory (style or content)
    if style_image:
        image_type = 'style'
    else:
        image_type = 'content'

    image_path = os.path.join(PATH, image_type, image_name)

    # Load the image from the link, convert it to RGB and store the image
    image = io.imread(link)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, image_rgb)

    update_dictionary(image_name, image_description, style_image)


def load_all_images(style_images, content_images):
    """
    Load all provided images.

    :param style_images: (list) list of style image loading information (link, image name, description, True)
    :param content_images: (list) list of content image loading information (link, image name, description, False)
    :return: None
    """

    all_images = style_images + content_images

    for image in all_images:

        assert len(image) == 4, 'The provided image information is incorrect. It must be of type: link (string), image_name (string), image_description (string), style_image (bool)'

        link, image_name, image_description, style_bool = image

        load_image_from_link(link, image_name, image_description, style_bool)


def test_preprocessing():
    """
    Test function to assess whether the conversion from image to tensor and back to image works. Throws
    assertions if something works incorrectly.

    :return: None
    """

    path_style = PATH_STYLE

    # Define path to test image
    img_path = os.path.join(path_style, 'the_scream.jpg')

    # Image to tensor
    img_tensor, img_shape = img2tensor(img_path)

    assert tf.is_tensor(img_tensor), 'Image is not a tensor.'

    # Tensor to image
    img = tensor2img(img_tensor, img_shape)

    assert isinstance(img, np.ndarray)


def prep_images(content_image_name, style_image_name):
    """
    Function to load and preprocess exactly one content and style image.

    :param content_image_name: (string) file name of the content image
    :param style_image_name: (string) file name of the style image
    return:
        image_content: (tensor) preprocessed content image
        image_style: (tensor) preprocessed style image
        image_content_shape: (array) original shape of the content image
    """

    # Load content image and store its original shape
    image_content, image_content_shape = img2tensor(os.path.join(PATH_CONTENT, content_image_name))

    # Load style image
    image_style, _ = img2tensor(os.path.join(PATH_STYLE, style_image_name))

    return image_content, image_style, image_content_shape


def clip_values(img):
    """
    Clips the values of a tensor image to the range of the respective color
    channel. This is necessary because of the zero-centering of each image in
    the preprocessing step.

    :param img: (tensor) tensor-image to clip the values
    return: (tensor) clipped version of tensor-image
    """

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    return tf.clip_by_value(img, clip_value_min=min_vals, clip_value_max=max_vals)

