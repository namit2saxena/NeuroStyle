import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cv2
import PIL
import os

from tensorflow.keras import Model

# setting up the optimizer
opt = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)


# For the loss function
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    gram_matrix_calc = tf.expand_dims(result, axis=0)
    input_shape = tf.shape(input_tensor)
    i_j = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return gram_matrix_calc / i_j


# Loading the weight to the model, setting the different layers
def load_vgg():
    vgg = tf.keras.applications.VGG19(include_top=True, weights=None)
    vgg.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels.h5')
    vgg.trainable = False
    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_output = vgg.get_layer(content_layers[0]).output
    style_output = [vgg.get_layer(style_layer).output for style_layer in style_layers]
    gram_style_output = [gram_matrix(output_) for output_ in style_output]

    model = Model([vgg.input], [content_output, gram_style_output])
    return model


# defining loss objects
def loss_object(style_outputs, content_outputs, style_target, content_target):
    style_weight = 1e-2
    content_weight = 1e-1
    content_loss = tf.reduce_mean((content_outputs - content_target) ** 2)
    style_loss = tf.add_n(
        [tf.reduce_mean((output_ - target_) ** 2) for output_, target_ in zip(style_outputs, style_target)])
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss


# preprocessing the images
def preprocess_imgs(content_image, style_image):
    vgg_model = load_vgg()
    content_target = vgg_model(np.array([content_image * 255]))[0]
    style_target = vgg_model(np.array([style_image * 255]))[1]
    return vgg_model, content_target, style_target;


# reading the images
def read_img(content_img_name, style_img_name):
    content_image = cv2.resize(cv2.imread(content_img_name), (224, 224))
    style_image = cv2.resize(cv2.imread(style_img_name), (224, 224))

    content_image = tf.image.convert_image_dtype(content_image, tf.float32)
    style_image = tf.image.convert_image_dtype(style_image, tf.float32)

    return content_image, style_image;


def train_step(image, epoch, vgg_model, content_target, style_target):
    with tf.GradientTape() as tape:
        output = vgg_model(image * 255)
        loss = loss_object(output[1], output[0], style_target, content_target)
    gradient = tape.gradient(loss, image)
    opt.apply_gradients([(gradient, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

    if epoch % 5 == 0:
        tf.print(f"Loss = {loss}")


def save_img(converted_image, dir_path, img_cnt):
    output_cnt = img_cnt
    tensor = converted_image * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    tensor = PIL.Image.fromarray(tensor)

    # cv2.imwrite assumes that the img given is BGR
    # PIL.Image.fromarray(img).save() assumes it is RGB
    file_name = "converted_{0}.jpg".format(output_cnt)

    #we are just converting this image to RGB from BGR and saving it using Pillow library
    im_rgb = cv2.cvtColor(np.array(tensor), cv2.COLOR_BGR2RGB)
    im_rgb = cv2.resize(im_rgb, (500, 300))
    image_path = os.path.join(dir_path, file_name)
    PIL.Image.fromarray(im_rgb).save(image_path)


def run_model(epochs_cnt, c_img, s_img, dir_path, img_cnt):
    # read the images first using cv2
    content_image, style_image = read_img(c_img, s_img)
    # process these images using vgg19
    vgg_model, content_target, style_target = preprocess_imgs(content_image, style_image)

    # we will input content_image, style_image, and the content_image as a variable
    # making the content_image variable so that this one gets edited to the final image
    image = tf.image.convert_image_dtype(content_image, tf.float32)
    image = tf.Variable([image])
    for i in range(epochs_cnt):
        train_step(image, i, vgg_model, content_target, style_target)
    save_img(image, dir_path, img_cnt)