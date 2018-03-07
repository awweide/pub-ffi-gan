#Code for reformating TIFF-formated Bjorkelangen data to
#data suitable for using with an Imagenet-based wgan-architeture
#Windows are selected from the source image and saved as separate
#image files in separate training and validation directories.
#Windows containing anomalous pixels, as per the masks in the
#data set, are omitted.

#window_size and stride are fully dynamic
#Some choice must be made as to how to deal with the
#different channels in the data set. A first minimum-effort
#solution is to simply pick out 3-channels and pretend we have
#a regular 3-channel color image.

import numpy as np
import pickle
import scipy.misc
import libtiff

window_size = 1
window_stride = 1
data_path ='/home/akeid/Hyper-data/Testdata-TIFF/Bjoerkelangen2008-09-24_22_rad_subs_20bands_sqrt.tif'
data_mask_outer_path = '/home/akeid/Hyper-data/Testdata-TIFF/outer_masks_Bjoerkelangen2008-09-24_22_subs.tif'
data_mask_inner_path = '/home/akeid/Hyper-data/Testdata-TIFF/inner_masks_Bjoerkelangen2008-09-24_22_subs.tif'

#Produces two 701*1600 numpy matrices
#image is a single channel of the image, such that we can treat the image as monochrome
#and feed it into a standard MNIST classifier
#mask has value 0 or 1 for each pixel, depending on whether the pixel is tagged as anomalous

tif = libtiff.TIFF.open(data_path, mode='r')
image_multichannel = tif.read_image()

#transpose from W,H,CHN to CHN,W,H
#rescale from 0,256 to -1,1
image_save = image_multichannel[:]

'''
Reduplicates a mostly-road window a large number of times in non-anomalous parts of the image
Ad hoc-implementation of up-weighted road
road_window = image_multichannel[175:225, 1200:1500, :]
road_x = 300
road_y = 50

for u in xrange(0, 701-road_y, road_y):
	for v in (900, 1200):
		image_multichannel[u:u+road_y, v:v+road_x, :] = road_window
'''

image_multichannel = image_multichannel.transpose([2,0,1])
image_multichannel = image_multichannel * (2/255.99) - 1


tif = libtiff.TIFF.open(data_mask_outer_path, mode='r')
mask_multichannel = tif.read_image()
mask_outer = np.sum(mask_multichannel, axis=0)

tif = libtiff.TIFF.open(data_mask_inner_path, mode='r')
mask_multichannel = tif.read_image()
mask_inner = np.sum(mask_multichannel, axis=0)

#Resamples image into an array of images, each a window of original image
#window_size is set to 64 to match with expected imagenet dimensions

window_images = []
window_labels = []
for i in xrange(0,image_multichannel.shape[1]-window_size,window_stride):
    for j in xrange(0,image_multichannel.shape[2]-window_size,window_stride):
        window_images.append(image_multichannel[:,i:i+window_size,j:j+window_size])
        window_mask_outer = mask_outer[i:i+window_size,j:j+window_size]
        window_mask_inner = mask_inner[i:i+window_size,j:j+window_size]
        label = -1
        if np.sum(window_mask_outer) > 0: label = 0
        if np.sum(window_mask_inner) > 0: label = 1
        window_labels.append(label)

#Encoding for labels:
#-1 : No anomalous pixels, neither inner nor outer (true background windows)
# 0 : Anomalous pixels, only outer (discard these as ambiguous)
# 1 : Anomalous pixels, even in inner (true anomalous windows)

window_images = np.array(window_images, dtype=np.float32)
window_labels = np.array(window_labels, dtype=np.int32)

#Split all images into those without and those with anomalies
#This code is awkward, but all non-awkward ways of doing it are non-simple
window_images_anomalous = [window_images[i] for i in xrange(len(window_images)) if window_labels[i] == 1] 
window_images_nonanomalous = [window_images[i] for i in xrange(len(window_images)) if window_labels[i] == -1]

#Splits anomaly free images into a train and validation set
#Seed ensures consistent splits
np.random.seed(0)
np.random.shuffle(window_images_nonanomalous)
split = np.array_split(window_images_nonanomalous, 5)
train, val = np.concatenate(split[0:4]), split[4]
anom = np.array(window_images_anomalous)

#Save images into different folders
#Each image is its own file
save_path = '/home/akeid/ffi-gan/data/single/'


pickle.dump(train, open(save_path + 'train.pkl', 'wb'))
pickle.dump(val, open(save_path + 'val.pkl', 'wb'))
pickle.dump(anom, open(save_path + 'anom.pkl', 'wb'))
print 'TRAIN: ', train.shape
print 'VAL: ', val.shape
print 'ANOM: ', anom.shape


#Best guess at selecting a RGB-view, i.e. that matches human color vision
#The selected bands are obtained by taking 'default bands' from metadata
#and dividing by 4 (80 channels are reduced to 20)
#This seems to correspond reasonably well with other sources

CHANNEL_VIEWS = [[7,5,1], [4,2,0], [11,9,7], [19,17,15]]
for name,view in zip(['0rgb', '1low', '2mid', '3hig'], CHANNEL_VIEWS):
	scipy.misc.imsave('real_image_{}.png'.format(name), image_save[:,:,view].astype(np.uint8), 'png')


scipy.misc.imsave('anomaly_mask.png', (mask_outer + mask_inner)*0.5, 'png')