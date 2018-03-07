"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave

def save_images(X, save_path):
    # [0, 1] -> [0,255]
    
    '''if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')
    '''
    
    #Replace [0,1]-conversion with [-1,1]-conversion
    #This makes more sense, but should never be used anyway;
    #Conversions happen in main function

    if isinstance(X.flatten()[0], np.floating):
        X = (255.99/2 * (X+1.0)).astype('uint8')


    #old views
    #channel_views = [[0,9,18],[1,3,5],[7,9,11],[13,15,17]]
    #channel_views = [[7,5,1], [4,2,0], [11,9,7], [18,9,0]]
    channel_views = [[7,5,1], [4,2,0], [11,9,7], [19,17,15]]

    #Used to select three channels to map to RGB for plotting
    #hyper-image as 2*2 mosaic of RGB,LOW,MID,HIGH 3-channel selections


    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    # BCHW -> BHWC
    X = X.transpose(0,2,3,1)
    h, w = X[0].shape[:2]
    img = np.zeros((3*h*nh, 3*w*nw, 3))

    for n, x in enumerate(X):
        j = 3*(n/nw)
        i = 3*(n%nw)

        img[j*h:j*h+h, i*w:i*w+w] = x[:,:,channel_views[0]]
        img[j*h+h:j*h+2*h, i*w:i*w+w] = x[:,:,channel_views[1]]
        img[j*h:j*h+h, i*w+w:i*w+2*w] = x[:,:,channel_views[2]]
        img[j*h+h:j*h+2*h, i*w+w:i*w+2*w] = x[:,:,channel_views[3]]

    imsave(save_path, img)
