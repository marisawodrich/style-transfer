import tensorflow as tf
from tensorflow.keras import Model

import numpy as np


class StyleModel(Model):

    def __init__(self, number_content_layers, layer_names):
        """
        Initializes a Style Transfer Model from layers of the VGG19 network.

        :param number_content_layers: (int) Number of used content layers
        :param layer_names: (list) list of strings with layer names (that must be layer names from VGG19)
        return: None
        """
        super(StyleModel, self).__init__()

        # Using the pretrained VGG19 model (on imagenet) with average pooling
        # layers as in the original paper.
        vgg19 = tf.keras.applications.VGG19(include_top=False,
                                            weights='imagenet',
                                            pooling='avg')

        # Store to access the correct layers later. All other layers will be
        # style layers
        self.number_content_layers = number_content_layers

        # Content and style layers respectively
        self.content_layers = [vgg19.get_layer(layer).output
                               for layer in layer_names[:number_content_layers]]
        self.style_layers = [vgg19.get_layer(layer).output
                             for layer in layer_names[number_content_layers:]]

        # Concatenate to get all output layers
        self.output_layers = self.content_layers + self.style_layers

        # Build model
        self.model = Model([vgg19.input], self.output_layers)


    def call(self, img):
        """
        Processes a given image with the Style Transfer model.

        :param img: (tensor) input image, zero-centered, BGR in range [0, 255]
        return: (list) list of outputs from each layer
        """

        # Get list of outputs
        outputs = self.model(img)

        return outputs


def gram_matrix(feature_maps, num_channels):
    """
    Defines the gram matrix for a given layer.

    :param feature_maps: feature maps from one layer
    :param num_channels: (int) number of feature maps in given layer
    return: gram-matrix of shape (num_channels x num_channels)
    """

    # Vectorize feature maps
    feature_maps_vectorized = tf.reshape(feature_maps, [-1, num_channels])

    # Multiply vectorized 'feature maps' to themselves to get the gram matrix
    # of shape (num_channels x num_channels).
    gram_matrix = tf.matmul(feature_maps_vectorized, feature_maps_vectorized, transpose_a=True)

    return gram_matrix


def loss_style(outputs, targets):
    """
    Calculates the style loss. The style loss is the weighted mean squared
    error between the output gram matrix and target gram matrix of each
    respective layer.

    :param outputs: layer activations for generated image
    :param targets: layer activations for target style image
    return: (float) style loss
    """

    # Initialize mean-squared-error (mse) loss.
    mse = 0.

    # Check if function was called with correct parameters.
    assert len(outputs) == len(targets), 'Outputs and targets must have the \
      same length, which refers to the same amount of layers.'

    # Iterate over all output and target feature maps.
    for output, target in zip(outputs, targets):
        assert np.array(output).shape == np.array(target).shape, 'Shape mismatch \
        of output and target feature maps.'

        # Get shape info: h -> height, w -> width, c -> number of channels
        _, h, w, c = np.array(output.shape).astype(np.uint32)

        # Calculate gram matrices
        gram_outputs = gram_matrix(output, c)
        gram_targets = gram_matrix(target, c)

        # Calculate error. Error is weighted first to get the mean error and then
        # by the number of style layers, which is represented by 'len(outputs)'.
        weighting_factor = 1 / (4. * c ** 2 * (h * w) ** 2)
        error = weighting_factor * \
                tf.reduce_sum(tf.square(gram_outputs - gram_targets))
        mse += (1. / len(outputs)) * error

    return mse


def loss_content(outputs, targets):
    """
    Calculates the mean squared error (MSE) for the content activations.

    :param outputs: layer activations of generated image
    :param targets: layer activations of target content image
    return: Mean squared error
    """

    assert len(outputs) == len(targets), 'Length of content outputs and targets must match!'

    # Weight by number of layers
    weight = 1 / len(outputs)

    # MSE is the mean over squared deviation of output and target.
    mse = 0
    for output, target in zip(outputs, targets):

        # Add up weighted errors
        mse += weight * tf.reduce_mean(tf.square(output - target))

    return mse
