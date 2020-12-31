import os
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras import backend as K
from PIL import Image, ImageOps


# import images from path
def import_images(path, num_images):
    image_buffer = []
    counter = 0
    for filename in os.listdir(path):
        if(num_images > counter):
            image = Image.open(path+filename)
            image_buffer.append(img_to_array(image))
            counter += 1
            if(counter % 100 == 0):
                print('number of imported images: ', counter)
    return image_buffer

# resizing with padding (white as default)
def image_resize_with_padding(path, image_out_size, num_images, fill=(255,255,255)):
  image_resized = []
  counter = 0
  for filename in os.listdir(path):
      if(counter < num_images):
          image = Image.open(path+filename)
          image = resize_with_padding(image, image_out_size, fill)
          image_resized.append(img_to_array(image))
          counter += 1
          if(counter % 100 == 0):
              print('number of imported images: ', counter)
  return image_resized

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def resize_with_padding(img, expected_size, fill):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=fill)

# plotting of images
def plotImages(images_arr, num_images):
    fig, axes = plt.subplots(1,num_images,figsize=(5,5))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# embedding for fusion layer
def inception_embedding(grayscaled_rgb, model):
    grayscaled_rgb_resized = []
    for image in grayscaled_rgb:
        image = resize(image, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(image)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    # embedding
    embed = model.predict(grayscaled_rgb_resized)
    return embed

# downsampling for color classification
def downsampling_rgb(images_rgb):
    images_rgb_resized = []
    for image in images_rgb:
        image = resize(image, (128, 128, 3), mode='constant')
        images_rgb_resized.append(image)
    images_rgb_resized = np.array(images_rgb_resized)
    return images_rgb_resized

def downsampling_l_channel(images_l_channel):
    images_l_channel_resized = []
    for image in images_l_channel:
        image = resize(image, (128, 128, 1), mode='constant')
        images_l_channel_resized.append(image)
    images_l_channel_resized = np.array(images_l_channel_resized)
    return images_l_channel_resized

def upsampling_ab_channel(images_ab_channel):
    images_ab_channel_resized = []
    for image in images_ab_channel:
        image = resize(image, (256, 256, 2), mode='constant')
        images_ab_channel_resized.append(image)
    images_ab_channel_resized = np.array(images_ab_channel_resized)
    return images_ab_channel_resized

# generator of batches (old versions)
'''def batch_generator(batch_size, DataGenerator, X, model):
    for batch in DataGenerator.flow(X, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = inception_embedding(grayscaled_rgb, model)
        batch_lab = rgb2lab(batch)
        Xbatch_lab = batch_lab[:,:,:,0]
        Ybatch_lab = batch_lab[:,:,:,1:] / 128
        yield ([Xbatch_lab.reshape(Xbatch_lab.shape+(1,)), embed], Ybatch_lab)
        '''
'''def batch_generator(batch_size, batches, model):
    for batch in train_batches:
        print(batch.shape)
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = inception_embedding(grayscaled_rgb, modelInceptionResNet)
        batch_lab = rgb2lab(batch)
        Xbatch_lab = batch_lab[:,:,:,0]
        Ybatch_lab = batch_lab[:,:,:,1:] / 128
        yield ([Xbatch_lab.reshape(Xbatch_lab.shape+(1,)), embed], Ybatch_lab)'''

# generator of batches without color classes      
def batch_generator(batch_size, batches, model):
    for batch in batches:
        grayscaled_rgb = gray2rgb(rgb2gray(batch[0][:][:][:]))
        embed = inception_embedding(grayscaled_rgb, model)
        batch_lab = rgb2lab(batch[0][:][:][:])
        Xbatch_lab = batch_lab[:,:,:,0]
        Ybatch_lab = batch_lab[:,:,:,1:] / 128
        yield ([Xbatch_lab.reshape(Xbatch_lab.shape+(1,)), embed], Ybatch_lab)

# generator of batches with color classes        
def batch_generator_color_classes(batch_size, batches, model, nn_search, sigma_nq, num_q, weight_q):
    for batch in batches:
        grayscaled_rgb = gray2rgb(rgb2gray(batch[0][:][:][:]))
        embed = inception_embedding(grayscaled_rgb, model)
        batch_lab = rgb2lab(batch[0][:][:][:])
        Xbatch_lab = batch_lab[:,:,:,0]

        # downsampling for ab-outputs
        batch_ds = downsampling_rgb(batch[0][:][:][:])
        batch_lab = rgb2lab(batch_ds)
        # reshaping
        nb, h, w, nc = batch_ds.shape
        num_row = nb * h * w
        reshaped_batch_ds = np.reshape(batch_lab[:,:,:,1:],(num_row,nc-1))

        # compute distances and indices of nearest neighbors
        dist_nq, idx_nq = nn_search.kneighbors(reshaped_batch_ds)

        # apply Gaussian kernel
        weights = np.exp(-dist_nq**2 / (2 * sigma_nq**2))
        weights = weights / np.sum(weights, axis=1)[:, np.newaxis]

        # apply softmax over num_nq
        Ybuffer = np.zeros((num_row, num_q))
        idx_pts = np.arange(num_row)[:, np.newaxis]
        Ybuffer[idx_pts, idx_nq] = weights

        # find qmax and apply class rebalancing
        id_qmax = np.argmax(Ybuffer,axis=-1)
        Ybatch_lab = np.reshape(weight_q[id_qmax] * Ybuffer,(nb,h,w,num_q))
        Ybatch_lab = np.reshape(Ybuffer,(nb,h,w,num_q))

        yield ([Xbatch_lab.reshape(Xbatch_lab.shape+(1,)), embed], Ybatch_lab)


def categorical_crossentropy_qclasses(y_true, y_pred):
    eps = 10e-18 # log regularization
    cross_ent = (-1) * K.mean(y_true * K.log(y_pred+eps))
    return cross_ent

# generate input data for model.evaluate
def evaluate_input(X_rgb,model):
    X_lab = rgb2lab(X_rgb)[:,:,:,0]
    X_lab = X_lab.reshape(X_lab.shape+(1,))
    grayscaled_rgb = gray2rgb(rgb2gray(X_rgb))
    embed = inception_embedding(grayscaled_rgb,model)
    Y_lab = rgb2lab(X_rgb)[:,:,:,1:]
    Y_lab = Y_lab / 128
    return [X_lab,embed], Y_lab
    
# PSNR error
def metric_psnr(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))