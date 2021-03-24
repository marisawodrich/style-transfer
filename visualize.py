import os
import matplotlib.pyplot as plt
from PIL import Image

from helpers import CONTENT_TITLE_DICT, STYLE_TITLE_DICT


def visualize_images(path, plot_title):
    """

    :param path:
    :param plot_title:
    :return:
    """

    # Test if path exists and is not empty.
    assert os.path.exists(path), 'Given path does not exist.'
    assert os.listdir(path), 'Given path is empty.'

    # Obtain all images from given directory.
    imgs = [element for element in os.listdir(path) if element.endswith('.jpg') or element.endswith('.png')]

    # Test if images were found, so plotting does not run into errors.
    assert imgs, 'No images found in given path. Please repeat the loading process.'

    # Special case for plotting when there is only one image in the folder.
    if len(imgs) == 1:
        _, ax = plt.subplots(1, figsize=(len(imgs) * 3, len(imgs) * 2))
    elif len(imgs) > 1:
        _, ax = plt.subplots(1, len(imgs), figsize=(len(imgs) * 3, len(imgs) * 2))

    # Define a dictionary that contains all content and style image names combined.
    title_dict = {**CONTENT_TITLE_DICT, **STYLE_TITLE_DICT}

    # Iterate over list of image references, load images and plot them beside each other.
    for i, img in enumerate(imgs):

        # If image name is listed in title dictionary, search for corresponding
        # image title. Else, just use the filename without the ending as title.
        if img in title_dict:
            img_name = title_dict[img]
        else:
            img_name = img.split('.')[0]

        # Load image
        img = Image.open(os.path.join(path, img))

        # Plot images. Special case for plotting if there is only one image in the given directory.
        if len(imgs) > 1:
            ax[i].imshow(img)
            ax[i].axis('off')
            ax[i].title.set_text(img_name)
        elif len(imgs) == 1:
            ax.imshow(img)
            ax.axis('off')
            ax.title.set_text(img_name)

    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.show()
