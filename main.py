import tensorflow as tf
import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt

from helpers import create_required_directories, load_all_images, test_preprocessing, choose_images, clip_values, \
    CONTENT_TITLE_DICT, STYLE_TITLE_DICT
from visualize import visualize_images
from model import build_model, loss_content, loss_style
from preprocessing import tensor_to_img

# Predefined style images:
#       - Starry Night by Vincent van Gogh
#       - The Scream by Edvard Munch
#       - Caféterrasse am Abend by Vincent van Gogh
#       - Keeping Busy in a Rizzy City by James Rizzy
STYLE_IMAGES = [['https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_'
                 'Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
                 'starry_night.jpg',
                 'Starry Night',
                 True],
                ['https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The'
                 '_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of'
                 '_Norway.jpg/300px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_'
                 'cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
                 'the_scream.jpg',
                 'The Scream',
                 True],
                ['https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Gogh4.jpg/300px-Gogh4.jpg',
                 'cafeterrasse_am_abend.jpg',
                 'Caféterrasse am Abend',
                 True],
                ['http://www.james-rizzi.com/wp-content/uploads/pictures/cache/2007_03_000_KeepingBusyIn'
                 'ARizziCity_800_600.jpg',
                 'rizzy.jpg',
                 'Rizzy Style',
                 True]]

# Predefined content images:
#       - Image of a sunflower
#       - Image of mountains behind a lake
#       - Image of a puppy
CONTENT_IMAGES = [['https://www.myhomebook.de/data/uploads/2020/02/gettyimages-1141659565-1024x683.jpg',
                   'sunflower.jpg',
                   'Sunflower',
                   False],
                  ['https://www.scinexx.de/wp-content/uploads/0/1/01-35131-nukliduhr01.jpg',
                   'mountain.jpg',
                   'Mountain',
                   False],
                  ['https://www.mera-petfood.com/files/_processed_/a/4/csm_iStock-521697453_7570f7a9b6.jpg',
                   'puppy.jpg',
                   'Puppy',
                   False]]


def transfer_artistic_style(model_st,
                            iterations,
                            content,
                            style,
                            optimizer,
                            weight_content,
                            weight_style,
                            save_name='final.jpg'):
    """
    Runs the artistic style transfer. The result image will be saved. It will contain the content of the
    content image and have the style of the style image. The 'strength' of content and style can be
    influenced by changing the respective weights.
    :param model_st: (tf sequential) Model for style transfer
    :param iterations: (int) Number of iterations optimization of image creation
    :param content: (tensor) image with content info
    :param style: (tensor) image with style info
    :param optimizer:
    :param weight_content: (float) weight for content image
    :param weight_style: (float) weight for style image
    :param save_name: (string) name for the result image (will be saved in directory 'images/generated'
    :return: None
    """

    print('Start')

    # Generate white image, result image should match the size of the
    # content image
    image_generated = tf.Variable(content, dtype=np.float32)

    # Get the content layers (first layer returned by model)
    target_content = model_st(content)[0]

    # Get the style layers ()
    target_style = model_st(style)[1:]

    print('Got target images, start optimizing.')

    for i in range(iterations):

        timer = time.time()

        # Compute gradients
        with tf.GradientTape() as tape:

            # Process image with model
            output = model_st(image_generated)

            output_content = output[0]
            output_style = output[1:]

            # tape.watch(image_generated)
            # Calculate loss with given weights
            content_loss = loss_content(output_content, target_content)
            style_loss = loss_style(output_style, target_style)
            loss = (weight_content * content_loss) + (weight_style * style_loss)

        # Gradients with respect to generated image
        gradients = tape.gradient(loss, image_generated)

        # Create a new image (apply gradients to image)
        optimizer.apply_gradients([(gradients, image_generated)])

        image_generated.assign(clip_values(image_generated))

        if i % 10 == 0:
            print('Iteration: ', str(i), '--- Loss: ', np.array(loss).astype(np.float32), '--- Content Loss: ',
                  np.array(content_loss).astype(np.float32), '--- Style Loss: ',
                  np.array(style_loss).astype(np.float32))
            print(time.time() - timer)

            img_gen = tensor_to_img(image_generated)

            # Show intermediate result
            #plt.imshow(img_gen)
            #plt.axis('off')
            #plt.show()

    # Save final image
    final_img = cv2.cvtColor(img_gen, cv2.COLOR_RGB2BGR)
    save_path = os.path.join('../images', 'generated')
    cv2.imwrite(os.path.join(save_path, save_name), final_img)


def run_all_combinations(model_st, content_images, style_images, content_weight, style_weight, num_iterations=200):
    """
    Creates artistic style transfer for all possible combinations of given content and style images. Images
    are saved in the directory 'images/generated' with the following naming convention:
            nameOfContentImage-nameOfStyleImage.jpg
    :param model_st: (tf sequential) Model for style transfer
    :param content_images: (list) list of names of content images
    :param style_images: (list) list of names of style images
    :param content_weight: (float) weight for the content image
    :param style_weight: (float) weight for the style image
    :param num_iterations: (int) number of iterations of optimization process
    :return: None
    """

    # Get all possible combinations of given content and style images.
    for name_content_img in content_images:

        for name_style_img in style_images:

            # Load images and preprocess them.
            content_img, style_img = choose_images(name_content_img, name_style_img)

            # Define name to save result image
            result_img_name = name_content_img.split('.')[0] + '-' + name_style_img

            transfer_artistic_style(model_st,
                                    num_iterations,
                                    content_img,
                                    style_img,
                                    optimizer,
                                    content_weight,
                                    style_weight,
                                    result_img_name)


if __name__ == '__main__':

    # Create directories from predefined paths.
    create_required_directories()

    # Here, one could append entries to content and style images

    # Load predefined images
    load_all_images(STYLE_IMAGES, CONTENT_IMAGES)

    # Visualize images
    visualize_images(os.path.join('../images', 'style'), 'STYLE IMAGES')
    visualize_images(os.path.join('../images', 'content'), 'CONTENT IMAGES')

    test_preprocessing()

    # Define layers for style transfer. The layers are obtained from VGG19,
    # pretrained on imagenet.

    # Second convolutional layer from the fifth block as content layer.
    content_layers = ['block5_conv2']

    # First convolutional layer from all five blocks as style layers.
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # Build model from those pretrained layers
    model, num_content_layers = build_model(content_layers, style_layers)

    # The model should not be trainable for the purpose of style transfer.
    model.trainable = False

    # Parameters: Adam optimizer, content and style weights, content and style images
    optimizer = tf.optimizers.Adam(learning_rate=0.03)
    weight_content = 1e4
    weight_style = 0.5

    # ---- EXECUTE ARTISTIC STYLE TRANSFER ----

    # Provide content and style images
    content_name = 'sunflower.jpg'
    style_name = 'the_scream.jpg'
    image_content, image_style = choose_images(content_name, style_name)

    # Define name to save final image
    save_name = content_name.split('.')[0] + '-' + style_name

    # Run artistic style transfer
    transfer_artistic_style(model, 20, image_content, image_style, optimizer, weight_content, weight_style, save_name)

    # ---- EXECUTE ARTISTIC STYLE TRANSFER ON ALL POSSIBLE COMBINATIONS ----

    # Get all content images
    all_content_images = CONTENT_TITLE_DICT.keys()

    # Get all style images
    all_style_images = STYLE_TITLE_DICT.keys()

    # Run style transfer on all possible content and style image combinations
    run_all_combinations(model, all_content_images, all_style_images, weight_content, weight_style, num_iterations=20)
