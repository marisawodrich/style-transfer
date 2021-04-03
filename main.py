import tensorflow as tf
import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt

from helpers import create_required_directories, load_all_images, test_preprocessing, prep_images, clip_values, \
    CONTENT_TITLE_DICT, STYLE_TITLE_DICT, PATH_GENERATED, PATH_STYLE, PATH_CONTENT
from visualize import visualize_images, plot_combinations, visualize_progress
from model import StyleModel, loss_content, loss_style
from processing import tensor2img

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


def transfer_artistic_style(iterations,
                            content,
                            style,
                            optimizer,
                            weight_content,
                            weight_style,
                            content_orig_shape=(224, 224, 3),
                            save_name='final.jpg',
                            visualize_intermediate=False):
    """
    This function runs the style transfer on a given content image with a given
    style image.

    :param iterations: (int) number of iterations of style transform
    :param content: (tensor) preprocessed content image
    :param style: (tensor) preprocessed style image
    :param optimizer: optimizer for model
    :param weight_content: (float) weight for the content loss
    :param weight_style: (float) weight for the style loss
    :param content_orig_shape: (array) shape of the original content image
    :param save_name: (string) name of the result image
    :param visualize_intermediate: (bool) whether to visualize during the transfer progress
    return: (list) list of strings containing the names of the intermediate images
    """

    print('Start')

    # Create an image to apply the operations to (like content image)
    image_generated = tf.Variable(content, dtype=np.float32)
    img_gen = tensor2img(image_generated, content_orig_shape)

    # Get the content layers (first layer returned by model)
    target_content = model(content)[:model.number_content_layers]

    # Get the style layers (all layers except the first for our case)
    target_style = model(style)[model.number_content_layers:]

    # Initialize list to store image names for plotting the progress later.
    intermediate_img_names = []

    print('Got target images, start optimizing.')

    start_time = time.time()
    time_intermediate = 0

    # Iteratively adapt the generated image to match the style image.
    for i in range(iterations):

        # Compute gradients
        with tf.GradientTape() as tape:

            # Process image with model
            output = model(image_generated)

            # Again, output of the first layer is content output, all other layers
            # are style outputs in our case
            output_content = output[:model.number_content_layers]
            output_style = output[model.number_content_layers:]

            # Calculate loss with given weights
            content_loss = loss_content(output_content, target_content)
            style_loss = loss_style(output_style, target_style)
            loss = (weight_content * content_loss) + (weight_style * style_loss)

        # Gradients with respect to generated image
        gradients = tape.gradient(loss, image_generated)

        # Generate optimized image (apply gradients to image)
        optimizer.apply_gradients([(gradients, image_generated)])

        # Assign new values image, but clip values before if necessary
        image_generated.assign(clip_values(image_generated))

        time_intermediate += time.time() - start_time - time_intermediate

        # Keep track of the progress. The progress information will be given
        # 20 times: some information and (optionally) an intermediate image
        # will be printed/ plotted.
        if i != 0 and int(i % (iterations / 20)) == 0:

            # Print information
            print('Iteration: ', str(i), '--- Time passed: ',
                  round(time_intermediate, 2), 'seconds --- Loss: ',
                  np.array(loss).astype(np.float32), '--- Content Loss: ',
                  np.array(content_loss).astype(np.float32), '--- Style Loss: ',
                  np.array(style_loss).astype(np.float32))

            # Convert tensor image back to 'normal' image
            img_gen = tensor2img(image_generated, content_orig_shape)

            # Convert to BGR, save image with cv2, and keep track of names of
            # intermediate images.
            intermediate_img = cv2.cvtColor(img_gen, cv2.COLOR_RGB2BGR)
            intermediate_img_names.append(str(i) + '.jpg')
            cv2.imwrite(os.path.join(PATH_GENERATED, str(i) + '.jpg'), intermediate_img)

            if visualize_intermediate:
                # Show intermediate result
                plt.imshow(img_gen)
                plt.axis('off')
                plt.show()

    # Save result image locally
    final_img = cv2.cvtColor(img_gen, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(PATH_GENERATED, save_name), final_img)

    # Print time for complete otimization process
    print('Finished', str(iterations), 'iterations in',
          round(time.time() - start_time, 2), 'seconds.')

    return np.array(intermediate_img_names)


def run_all_combinations(content_images,
                         style_images,
                         optimizer,
                         weight_content,
                         weight_style,
                         num_iterations=1000):
    """
    Runs style transfer on all possible combinations of given content and
    style images.

    :param content_images: (list) list of strings of content image file names
    :param style_images: (list) list of strings of style image file names
    return: (list) list of strings with result image names
    """
    save_names = []

    # Iterate over content image
    for content_name in content_images:

        # For each content image, get the combination with each style image
        for style_name in style_images:
            # Keep track of progress
            print('Next image...')

            # Load and preprocess images
            image_content, image_style, image_content_shape = prep_images(content_name, style_name)

            # Define name to save result image
            save_name = content_name.split('.')[0] + '-' + style_name
            save_names.append([content_name, style_name, save_name])

            # Run artistic style transfer
            transfer_artistic_style(num_iterations,
                                    image_content,
                                    image_style,
                                    optimizer,
                                    weight_content,
                                    weight_style,
                                    image_content_shape,
                                    save_name)

    return save_names


if __name__ == '__main__':

    # Create directories from predefined paths.
    create_required_directories()

    # Here, one could append entries to content and style images

    # Load predefined images
    load_all_images(STYLE_IMAGES, CONTENT_IMAGES)

    # Visualize images
    visualize_images(PATH_STYLE, 'STYLE IMAGES')
    visualize_images(PATH_CONTENT, 'CONTENT IMAGES')

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

    all_layers = content_layers + style_layers

    # Build model from those pretrained layers
    model = StyleModel(len(content_layers), all_layers)

    # Model should not be trainable for the purpose of style transfer
    for layer in model.layers:
        layer.trainable = False

    # Parameters: Adam optimizer, content and style weights, content and style images
    optimizer = tf.optimizers.Adam(learning_rate=4)
    weight_content = 1
    weight_style = 1e-2

    # ---- EXECUTE ARTISTIC STYLE TRANSFER ----

    # Provide content and style images
    content_name = 'sunflower.jpg'
    style_name = 'the_scream.jpg'
    image_content, image_style, image_content_shape = prep_images(content_name, style_name)

    # Define name to save final image
    save_name = content_name.split('.')[0] + '-' + style_name

    # Run artistic style transfer
    img_names = transfer_artistic_style(20,
                                        image_content,
                                        image_style,
                                        optimizer,
                                        weight_content,
                                        weight_style,
                                        image_content_shape,
                                        save_name,
                                        visualize_intermediate=False)

    # Run visualizing code
    visualize_progress(img_names)

    # ---- EXECUTE ARTISTIC STYLE TRANSFER ON ALL POSSIBLE COMBINATIONS ----

    # Get all content images
    all_content_images = CONTENT_TITLE_DICT.keys()

    # Get all style images
    all_style_images = STYLE_TITLE_DICT.keys()

    # Run style transfer on all those possible combinations
    save_names_own_content_imgs = run_all_combinations(all_content_images,
                                                       all_style_images,
                                                       optimizer,
                                                       weight_content=1,
                                                       weight_style=1e-2,
                                                       num_iterations=10)

    # Visualize all combinations
    plot_combinations(save_names_own_content_imgs)
