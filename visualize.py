import os
import matplotlib.pyplot as plt
from PIL import Image

from helpers import CONTENT_TITLE_DICT, STYLE_TITLE_DICT, PATH_CONTENT, PATH_STYLE, PATH_GENERATED


def visualize_images(path, plot_title, notebook=False):
    """
    This function creates a plot of all images contained in the directory (given
    path).

    :param path: (string) path to directory of all images to be plotted
    :param plot_title: (string) title of the plot
    return: None
    """

    assert os.path.exists(path), 'Given path does not exist.'
    assert os.listdir(path), 'Given path is empty.'

    path_content = os.listdir(path)

    # Remove files that are not images (This can be seen as a step of caution
    # but usually path_content and imgs should not differ because we only load
    # images to the folder).
    imgs = [element for element in path_content if element.endswith('.jpg') or \
            element.endswith('.png')]

    # Test if images were found, so plotting does not run into errors.
    assert imgs, 'No images found in given path. Please repeat the loading process.'

    if len(imgs) > 1:
        _, ax = plt.subplots(1, len(imgs), figsize=(len(imgs) * 3, len(imgs) * 2))
    elif len(imgs) == 1:
        _, ax = plt.subplots(1, figsize=(len(imgs) * 3, len(imgs) * 2))

    # Define a dictionary that contains all content and style image names combined.
    title_dict = {**CONTENT_TITLE_DICT, **STYLE_TITLE_DICT}

    # Iterate over list of image references, load images
    # and plot them beside each other.
    for i, img in enumerate(imgs):

        # If image name is listed in title dictionary, search for corresponding
        # image title. Else, just use the filename without the ending as title.
        if img in title_dict:
            img_name = title_dict[img]
        else:
            img_name = img.split('.')[0]

        # Load image
        img = Image.open(os.path.join(path, img))

        # Plot images (Special case for plotting if there is only one image in the
        # given path.)
        if len(imgs) > 1:
            ax[i].imshow(img)
            ax[i].axis('off')
            ax[i].title.set_text(img_name)
        elif len(imgs) == 1:
            ax.imshow(img)
            ax.axis('off')
            ax.title.set_text(img_name)

    # Running this locally and not in the demo.ipynb, one has to close the created plot
    # manually before the code execution continues
    if notebook:
        print(plot_title)
    else:
        plot_title += '\n(Code execution pauses until this window is closed)'
        plt.suptitle(plot_title)

    plt.tight_layout()
    plt.show()


def visualize_progress(image_names):
    """
    Visualizes progress of style transfer for the latest combination.
    (Images in generated image folder will be overwritten in each run to prevent
    memory issues.)

    :param image_names: (list) list of strings with all image names of intermediate images
    return: None
    """

    # Path to generated images
    gen_path = PATH_GENERATED

    # Iterate over all saved image names.
    for img_name in image_names:

        # Load image from directory of generated images
        img = plt.imread(os.path.join(gen_path, img_name))


        # Show image
        plt.imshow(img)
        plt.axis('off')
        plt.title('Iteration: ' + img_name.split('.')[0])
        plt.show(block=False)

        # Wait after each plot and clear old plot to get progress plot
        plt.pause(1)
        plt.close()


def plot_combinations(save_names):
    """
    Plots all combinations and the original images.

    save_names: (list) list of list that contains for each combination
        the name of the content image, name of the style image and name of
        combined images (transferred style)
    return: None
    """

    # Iterate over all result image names
    for save_name in save_names:
        # Load images from their directory
        img_content = plt.imread(os.path.join(PATH_CONTENT, save_name[0]))
        img_style = plt.imread(os.path.join(PATH_STYLE, save_name[1]))
        img_combined = plt.imread(os.path.join(PATH_GENERATED, save_name[2]))

        _, ax = plt.subplots(1, 3, figsize=(8, 2))

        # Plot images
        ax[0].imshow(img_content), ax[0].axis('off')
        ax[1].imshow(img_style), ax[1].axis('off')
        ax[2].imshow(img_combined), ax[2].axis('off')

        title_dict = {**CONTENT_TITLE_DICT, **STYLE_TITLE_DICT}

        plt.suptitle(title_dict[save_name[0]] + ' + ' + title_dict[save_name[1]])
        plt.show()
