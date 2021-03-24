import tensorflow as tf
from tensorflow.keras import Model

import numpy as np

def build_model(content_layers, style_layers):
    """

    :param content_layers:
    :param style_layers:
    :return:
    """

    num_content_layers = len(content_layers)
    layers_combined = content_layers + style_layers

    # Load pretrained VGG19 model with average pooling.
    vgg19 = tf.keras.applications.VGG19(include_top=False,
                                        weights='imagenet',
                                        pooling='avg')

    # The model should not be trainable for the purpose of style transfer.
    vgg19.trainable = False

    # Get outputs from the content and style layers we defined above
    output = [vgg19.get_layer(layer).output for layer in layers_combined]

    # Build the model
    model = Model([vgg19.input], output)

    return model, num_content_layers

def gram_matrix(filter_response, num_channels):
    """

    :param filter_response:
    :param num_channels:
    :return:
    """

    # Reshape c-many to vectors
    filter_response = tf.reshape(filter_response, [-1, num_channels])   # hier bin ich auch unsicher

    # Multiply 'filters' to itself.
    gram_matrix = tf.matmul(filter_response, filter_response, transpose_a=True)  # unsicher welches ich true und welches false setzen muss # JOHANNA: Hier könnten wir zur Kontrolle einfach einmal schauen, ob wir die shape NxN (wie im paper) rausbekommen :)

    # Divide by the number of channels for ... .
    gram_matrix = gram_matrix / tf.cast(num_channels, tf.float32)   # hier 1. unsicher ob das die richtigen Variablen sind und 2. ob ich das mit normaler Division machen kann oder auch eine tf function brauche (wie tf.matmul...)

    return gram_matrix


def loss_style(outputs, targets):
    """

    :param outputs:
    :param targets:
    :return:
    """

    # Initialize mean-squared-error (mse) loss.
    mse = 0

    # Check if function was called with correct parameters. # TODO: Further assertions
    assert len(outputs) == len(targets), 'Outputs and targets must have the same length, ' \
                                         'which refers to the same amount of layers.'

    # Iterate over all output and target feature maps.
    for output, target in zip(outputs, targets):
        assert np.array(output).shape == np.array(target).shape, 'Shape mismatch of output and target feature maps.'

        # Get shape info: h -> height, w -> width, c -> number of channels
        _, h, w, c = np.array(output.shape).astype(np.uint32)

        # Calculate gram matrices
        gram_outputs = gram_matrix(output, c)
        gram_targets = gram_matrix(target, c)

        # Calculate error. Error is immediately weighted by the number of style
        # layers, which is represented by 'len(outputs)'.
        error = (1 / len(outputs)) * tf.square(gram_outputs - gram_targets)
        mse += tf.reduce_mean(error)

        # DEVIATION FROM REPLICATION PAPER:
        # In the original paper, 'mse' would be divided by '(4 * c**2 * (h*w)**2)'.
        # We found that for our implementation, this makes the loss extremely
        # small. We could scale this value up with a very high weight for the style
        # loss, however we found our results to be very appealing without this
        # term, so we simply left it out in our implementation but still wanted to
        # mention it.

    return mse


def loss_content(output, target):
    """

    :param output:
    :param target:
    :return:
    """

    # MSE loss
    error = tf.square(output - target)
    mse = 1/2 * tf.reduce_mean(error)

    return mse
