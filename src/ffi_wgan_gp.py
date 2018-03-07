import os, sys
sys.path.append(os.getcwd())
import time

import numpy as np
import tensorflow as tf
from itertools import islice

import tflib as lib
import tflib.save_images
import tflib.ffi_bjrkl
import tflib.plot

import pickle
import libtiff
import scipy.misc

#Heavily modified code, based on https://github.com/igul222/improved_wgan_training

#Exposes NOISE_DIM, NOISE_WEIGHT, DIM, MINIMIZE_ITERATIONS, MINIMIZE_RUNS to command line arguments
arg1 = int(sys.argv[1])
arg2 = float(sys.argv[2])
arg3 = int(sys.argv[3])
arg4 = int(sys.argv[4])
arg5 = int(sys.argv[5])

DATA_DIR = '/home/akeid/ffi-gan/data/single'
MODEL_PATH = ''
    #Insert model path to load model model and skip training
    #If left as empty string '', trains model from scratch
    #Default behavior saves model every EVAL_INTERVAL - these can be loaded

MODE = 'wgan-gp' #'wgan-lp'
DIM = arg3 # Model dimensionality - higher value <-> increased model complexity
    #Default 64 - DIM, image size and network design determine memory and time intensity
NOISE_DIM = arg1 # Defaults to 128 in original implementation - used to be hard-coded

CRITIC_ITERS = 5 # Defaults to 5 in original implementation - Discriminator iterations per Generator iteration
N_GPUS = 4 # Number of GPUs

BATCH_SIZE = 256 # Batch size. Must be a multiple of N_GPUS
ITERS = 20000    # How many iterations to train for : Default : 200000

EVAL_INTERVAL = 5000    #Log/plot/validate every EVAL_INTERVAL iterations
EVAL_BATCHES = 1        #This feeds validation and anomaly sets through the discriminator
                        #Note: EVAL is time intensive

MINIMIZE_RUNS = arg5        #Uses the best results from across this amount of runs
                            #Helps avoid convergence / local minima issues
                            #1 is a valid value
MINIMIZE_ITERATIONS = arg4  #Minimization iterations for each run - GradientDescent on noise for reconstruction cost
NOISE_WEIGHT = 10**-arg2    #Scales the relative importance of pixel-wise similarity
if arg2 == 100: NOISE_WEIGHT = 0

#Dynamic learning rate values - makes very little difference
TRAIN_RATE_BOUNDARIES = [(ITERS / 3) * _ for _ in xrange(1,3)]
TRAIN_RATE_VALUES = [5e-6, 1e-6, 5e-7]

LAMBDA = 10 # Gradient penalty lambda hyperparameter
CHANNELS = 20 # Number of spectral channels in dataset
CHANNEL_VIEWS = [[7,5,1], [4,2,0], [11,9,7], [19,17,15]]

#Stride and kernels for conv3d and conv3d_transpose layers
#KERNEL and STRIDE resamples by a factor 2 in spectral domain - STRIDE_SPECIAL handles sampling 5<->4
KERNEL = (2,1,1)
STRIDE, STRIDE_SPECIAL = (2,1,1), (1,1,1)

WINDOW = 1 # Width = height of windows
OUTPUT_DIM = WINDOW*WINDOW*CHANNELS # Number of pixels in each iamge
    #This must match our image dimensionality - window_size and number_of_channels
    #Use declared constant instead of magic constant for channels
MAX_UINT8 = 255.99  #Used for rescaling and avoiding modulus issues
LOG_EPSILON = 1e-9  #Added to avoid log(0)-errors


def batch_cost(batch_real, batch_fake, batch_noise):
    #Partitions data into different chunks for each GPU : GPU-parallelization
    split_batch_real = tf.split(batch_real, len(DEVICES))
    split_batch_fake = tf.split(batch_fake, len(DEVICES))
    split_batch_noise = tf.split(batch_noise, len(DEVICES))
    
    #Collector for partial results from each GPU
    cost_collector = []

    #Parallelized cost calculations
    for device_index, (device, _real, _fake, _noise) in enumerate(zip(DEVICES, split_batch_real, split_batch_fake, split_batch_noise)):
        with tf.device(device), tf.name_scope('device_index'):
            #Assuming real and fake images independent and with uniformly distributed channel-pixel values in [-1,1]
            #The expected batch_diff per channel-pixel is 2/3=0.67
            #If distributions are instead normal, the value is somewhere around 4/3 = 1.33
            #And for a non-uniform, normal distribution truncated to [-1,1] it is 1.0

            #Furthermore, the expected batch_noise per NOISE_DIM is 1.0

            #By dividing by OUTPUT_DIM and NOISE_DIM respectively, we normalize batch_diff and batch_noise to roughly 1.0
            #NB! - for the case where _real and _fake are independent, i.e. the Generator is clueless
            #The terms must be weighted depending on how closely the fake image can be expected to match the real image

            #Failure mode 1: NOISE_WEIGHT too large, minimization collapses to the most likely all-zero noise ignoring image similarity
                #Particularly likely when OUTPUT_DIM >>> NOISE_DIM
            #Failure mode 2: NOISE_WEIGHT too small, minimization recreates image with highly unlikely noise, failing to discriminate
            #likely from unlikely noise because noise-term is drowned out by tiny pixel-differences
                #Particularly likely when NOISE_DIM >~= OUTPUT_DIM

            batch_diff = tf.subtract(_real, _fake)
            batch_diff_norm = tf.reduce_sum(tf.square(batch_diff), axis=[1,2,3]) / OUTPUT_DIM
            
            batch_noise_norm = tf.reduce_sum(tf.square(_noise), axis=[1]) / NOISE_DIM
            cost_collector.append(batch_diff_norm + NOISE_WEIGHT * batch_noise_norm)

    #Return collected results concat'ed into shape=[BATCH_SIZE]
    return tf.concat(cost_collector, 0)

def batch_minimize(batch_real):
    #Do minimzation a number of times with different initial values
    for min_run in xrange(MINIMIZE_RUNS):
        #Initialize gen_input with random noise and reset optimizer
        _,_ = session.run([init_op_noise, init_op_minimizer])    
        #Minimize noise for a number of iterations
        for min_iter in xrange(MINIMIZE_ITERATIONS):
            _, batch_scores_run, batch_fake_run = session.run([minimize_op, batch_reconstruction_cost, gen_from_noise], feed_dict = {batch_real_data : batch_real})
        #If first run, no comparisons to make
        if min_run == 0:
            batch_scores_min = batch_scores_run
            batch_fake_min = batch_fake_run
        #Otherwise, keep the lowest score and corresponding image for each
        else:
            m = np.argmin([batch_scores_min, batch_scores_run], axis=0)
            for i in xrange(BATCH_SIZE):
                if m[i]: batch_scores_min[i],batch_fake_min[i] = batch_scores_run[i],batch_fake_run[i]
            #Unlooped version - no obvious improvement in performance
            #batch_scores_min = [[batch_scores_min[i], batch_scores_run[i]][v] for i,v in enumerate(m)]
            #batch_fake_min = [[batch_fake_min[i], batch_fake_run[i]][v] for i,v in enumerate(m)]

    return batch_scores_min, batch_real, batch_fake_min

def anomaly_confusion_matrix(val_scores, anom_scores):
    #Note: val_scores and anom_scores not equal length
    v_mean, a_mean = np.mean(val_scores), np.mean(anom_scores)
    threshhold = np.sort(np.concatenate([val_scores, anom_scores],axis=0))[len(val_scores)]

    TP = len(anom_scores[anom_scores < threshhold])
    FP = len(anom_scores) - TP
    TN = len(val_scores[val_scores > threshhold])
    FN = len(val_scores) - TN
    
    return np.array((TP,FP,TN,FN))+0.0

def scale_range(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input
#Non-destructive scaler
def scale_uint8(input): return scale_range(np.array(input,copy=True), 0.0, MAX_UINT8)

def end_validation():
    start_time = time.time()
    print 'Full heatmaps being generated'

    #Load full image
    tif = libtiff.TIFF.open('/home/akeid/Hyper-data/Testdata-TIFF/Bjoerkelangen2008-09-24_22_rad_subs_20bands_sqrt.tif', mode='r')
    image_multichannel = tif.read_image()
    image_multichannel = image_multichannel * (2/255.99) - 1
    y,x = image_multichannel.shape[:2]

    #load outer and inner masks
    tif = libtiff.TIFF.open('/home/akeid/Hyper-data/Testdata-TIFF/outer_masks_Bjoerkelangen2008-09-24_22_subs.tif', mode='r')
    image_outer = tif.read_image()
    tif = libtiff.TIFF.open('/home/akeid/Hyper-data/Testdata-TIFF/inner_masks_Bjoerkelangen2008-09-24_22_subs.tif', mode='r')
    image_inner = tif.read_image()

    #image_inner and image_outer are image-sized masks, 0 outside of ROIs, value = id inside of ROI
    max_id = image_outer.shape[0]
    for i in xrange(max_id):
        image_outer[i,:,:] *= (i+1)
        image_inner[i,:,:] *= (i+1)
    image_outer = np.sum(image_outer, axis=0)
    image_inner = np.sum(image_inner, axis=0)

    #Generate heatmap of entire image with discriminator
    image_heatmap_disc = np.zeros_like(image_multichannel[:,:,0])
    image_heatmap_gen = np.zeros_like(image_multichannel[:,:,0])
    image_reconstruction = np.zeros_like(image_multichannel)

    for u in xrange(0,y):
        #Display progress in heavy loop
        if u % 100 == 0: print 'Full recreation loop: {} of {} done'.format(u,y)
        for v in xrange(0,x-x%BATCH_SIZE,BATCH_SIZE):
            batch_windowed_images = image_multichannel[u,v:v+BATCH_SIZE,:]
            batch_windowed_images = batch_windowed_images.reshape([BATCH_SIZE, CHANNELS, WINDOW, WINDOW])
            image_heatmap_disc[u,v:v+BATCH_SIZE] = session.run(disc_real, feed_dict={batch_real_data: batch_windowed_images})
            bm_score, _, bm_fake = batch_minimize(batch_windowed_images)
            image_heatmap_gen[u,v:v+BATCH_SIZE] = bm_score
            image_reconstruction[u,v:v+BATCH_SIZE,:] = bm_fake.reshape([BATCH_SIZE, CHANNELS])

    #Flips discriminator scores so they have same interpretation as recreation loss
    image_heatmap_disc *= -1
    #Shifts discriminator scores so that they are non-negative
    image_heatmap_disc -= np.min(image_heatmap_disc)

    image_reconstruction = (1+image_reconstruction) * (255.99/2)
    image_multichannel = (1+image_multichannel) * (255.99/2)
    
    scipy.misc.imsave('discriminator_heatmap.png', scale_uint8(image_heatmap_disc), 'png')
    scipy.misc.imsave('discriminator_heatmap_log.png', scale_uint8(np.log10(image_heatmap_disc+LOG_EPSILON)), 'png')
    scipy.misc.imsave('generator_heatmap.png', scale_uint8(np.log10(image_heatmap_gen+LOG_EPSILON)), 'png')
    scipy.misc.imsave('mixed_heatmap.png',
        0.5*scale_uint8(np.log10(image_heatmap_disc+LOG_EPSILON))+0.5*scale_uint8(np.log10(image_heatmap_gen+LOG_EPSILON)), 'png'
        )
    
    pickle.dump(image_heatmap_disc, open('heatmap_disc.pkl', 'wb'))
    pickle.dump(image_heatmap_gen, open('heatmap_gen.pkl', 'wb'))
    pickle.dump(image_reconstruction, open('image_reconstruction.pkl', 'wb'))

    
    for name,view in zip(['0rgb', '1low', '2mid', '3hig'], CHANNEL_VIEWS):
        scipy.misc.imsave('generator_reconstruction_{}.png'.format(name), (image_reconstruction[:,:,view]).astype(np.uint8), 'png')
        scipy.misc.imsave('generator_reconstruction_diff_signed_{}.png'.format(name), 
            scale_uint8((image_reconstruction[:,:x-x%BATCH_SIZE,view]-image_multichannel[:,:x-x%BATCH_SIZE,view])), 'png'
            )
        scipy.misc.imsave('generator_reconstruction_diff_unsigned_{}.png'.format(name), 
            scale_uint8(np.abs(image_reconstruction[:,:x-x%BATCH_SIZE,view]-image_multichannel[:,:x-x%BATCH_SIZE,view])), 'png'
        )

    reconstruction_cost_by_id = [[] for i in xrange(max_id+1)]
    discrimination_cost_by_id = [[] for i in xrange(max_id+1)]
    for u in xrange(0,y):
        for v in xrange(0,x-BATCH_SIZE):
            if image_outer[u,v] != 0: 
                id = image_inner[u,v]
                if id != 0:
                    reconstruction_cost_by_id[id].append(image_heatmap_gen[u,v])
                    discrimination_cost_by_id[id].append(image_heatmap_disc[u,v])
            else:
                reconstruction_cost_by_id[0].append(image_heatmap_gen[u,v])
                discrimination_cost_by_id[0].append(image_heatmap_disc[u,v])

    for i in xrange(1, max_id+1):
        lib.plot.ROC_clean_log('ROC_fullbg_id_{}'.format(i),
            (discrimination_cost_by_id[0],discrimination_cost_by_id[i]),
            (reconstruction_cost_by_id[0],reconstruction_cost_by_id[i])
            )

    lib.plot.ROC_clean_log('ROC_fullbg_all',
        (discrimination_cost_by_id[0], np.concatenate(discrimination_cost_by_id[1:])),
        (reconstruction_cost_by_id[0], np.concatenate(reconstruction_cost_by_id[1:]))
        )
    
    lib.plot.hist_disc(np.array(discrimination_cost_by_id[0]), np.concatenate(discrimination_cost_by_id[1:]))
    lib.plot.hist_gen(np.array(reconstruction_cost_by_id[0]), np.concatenate(reconstruction_cost_by_id[1:]))

    print 'Full heatmaps finished: ', time.time() - start_time

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
    #GPU-naming

#Helper function for repeated tensor->normalize(tensor)->activate(tensor) operations
def NormalizeThenActivate(tensor, normalizer=tf.identity, activator=tf.identity):
    #Normalization removed after theoretical considerations - uncomment to renable
    #tensor = normalizer(tensor)
    tensor = activator(tensor)
    return tensor

#Pure Tensorflow implementation of 3D-convoluting ResNet
def P3dDiscriminator(inputs, dim=DIM):
    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        norm = tf.contrib.layers.layer_norm
        act = tf.nn.relu
        #Reshape input into [BATCH/N_GPUS, IMAGE (spectral resolution,x resolution, y resolution), 1 (model dimensionality)]
        #This reshape is mostly superfluous - triggers for gradient penalty calculation
        output = tf.reshape(inputs, [-1, CHANNELS, WINDOW, WINDOW, 1])
        output = tf.layers.conv3d(output, 1*dim, KERNEL, padding='same')
        
        def DiscResBlock(input, input_dim, output_dim, kernel, stride):
            shortcut = tf.layers.conv3d(input, output_dim, kernel, stride, padding='valid')
            conv = input
            conv = NormalizeThenActivate(conv, normalizer=norm, activator=act)
            conv = tf.layers.conv3d(conv, input_dim, kernel, padding='same')
            conv = NormalizeThenActivate(conv, normalizer=norm, activator=act)
            conv = tf.layers.conv3d(conv, output_dim, kernel, stride, padding='valid')
            return shortcut + conv

        output = DiscResBlock(output, 1*dim, 2*dim, kernel=KERNEL, stride=STRIDE)
        output = DiscResBlock(output, 2*dim, 4*dim, kernel=KERNEL, stride=STRIDE)
        output = DiscResBlock(output, 4*dim, 8*dim, kernel=KERNEL, stride=STRIDE_SPECIAL)
        output = DiscResBlock(output, 8*dim, 16*dim, kernel=KERNEL, stride=STRIDE)
        output = DiscResBlock(output, 16*dim, 32*dim, kernel=KERNEL, stride=STRIDE)

        #Collapse all per-sample resolution into a single vector
        #32*dim dimensionality ("filters") and 1*2*2 spectral-spatial dimensionality
        output = tf.reshape(output, [-1, 32*dim])
        output = tf.layers.dense(output, 1)
        #output shaped as n_samples
        return tf.reshape(output, [-1])

def P3dGenerator(n_samples, noise=None, dim=DIM, is_training=True):
    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        norm = lambda x: tf.layers.batch_normalization(x, momentum=0.999, training=is_training)  
        act = tf.nn.relu
        #Generate noise to seed the generator with
        #If noise passed as argument, use this instead
        #This allows us to re-use the same noise across iterations
        #for visualizing images generated by the Generator
        #Introduce NOISE_DIM constant (length of noise vector)
        if noise is None: noise = tf.random_normal([n_samples, NOISE_DIM], dtype=tf.float32)
        
        output = tf.layers.dense(noise, 16*dim)

        #Here we need to reshape into 5D in order to work with conv3d
        #Tensorflow expects 'NDHWC' format - n_samples,depth,height,width,channels
        #We map 'd' to wavelength
        output = tf.reshape(output, [-1, 1, 1, 1, 16*dim])

        #We have to make custom Residual blocks for our conv3d-case
        #[spec, spat_x, spat_y] dimensions
        #[2,1,1] <-> [4,2,2] <-> [5,3,3] <-> [10,4,4] <-> [20,5,5]

        def GenResBlock(input, input_dim, output_dim, kernel, stride):
            #shortcut = Upsample3d(input) - looks like nonsense
            shortcut = tf.layers.conv3d_transpose(input, output_dim, kernel, stride, padding='valid')
            conv = input
            conv = NormalizeThenActivate(conv, normalizer=norm, activator=act)
            conv = tf.layers.conv3d_transpose(conv, output_dim, kernel, padding='same')
            conv = NormalizeThenActivate(conv, normalizer=norm, activator=act) 
            conv = tf.layers.conv3d_transpose(conv, output_dim, kernel, stride, padding='valid')
            return shortcut + conv

        output = GenResBlock(output, 32*dim, 16*dim, kernel=KERNEL, stride=STRIDE)
        output = GenResBlock(output, 16*dim, 8*dim, kernel=KERNEL, stride=STRIDE)
        output = GenResBlock(output, 8*dim, 4*dim, kernel=KERNEL, stride=STRIDE_SPECIAL)
        output = GenResBlock(output, 4*dim, 2*dim, kernel=KERNEL, stride=STRIDE)
        output = GenResBlock(output, 2*dim, 1*dim, kernel=KERNEL, stride=STRIDE)

        output = NormalizeThenActivate(output, normalizer=norm, activator=act)
        output = tf.layers.conv3d_transpose(output, 1, KERNEL, padding='same')
        output = tf.nn.tanh(output)
     
        return tf.reshape(output, [-1, CHANNELS, WINDOW, WINDOW])

Generator, Discriminator = P3dGenerator, P3dDiscriminator

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    #Placeholder for full-batch (not split for GPU-parallels) image data
    batch_real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, CHANNELS, WINDOW, WINDOW])

    #Counter for learning rate and tf variable for decaying learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.piecewise_constant(global_step, TRAIN_RATE_BOUNDARIES, TRAIN_RATE_VALUES, name=None)

    #Partitions data into different chunks for each GPU : GPU-parallelization
    split_real_data = tf.split(batch_real_data, len(DEVICES))
   
    gen_costs, disc_costs, disc_costs_gp, disc_reals, fake_datas = [],[],[],[],[]
    #for index of gpu : real-data partition assigned to this gpu
    for device_index, (device, real_data) in enumerate(zip(DEVICES, split_real_data)):
        with tf.device(device), tf.name_scope('device_index'):
            #This is the work-loop parallelized for each GPU

            #Reshape should not be necessary - left as a sanity check
            #Will produce runtime errors if dimensions do not match
            real_data = tf.reshape(real_data, [BATCH_SIZE/len(DEVICES), CHANNELS, WINDOW, WINDOW])

            #Uses the generator to generate fake images
            #The total batch-size is distribued across all GPU devices
            with tf.name_scope('G_main'): fake_data = Generator(BATCH_SIZE/len(DEVICES))

            #Applies to discriminator to both the real_data from data set and fake_data from generator
            with tf.name_scope('D_real'): disc_real = Discriminator(real_data)
            with tf.name_scope('D_fake'): disc_fake = Discriminator(fake_data)

            #gen_cost is the negative mean of the discriminator output for the fake data
            #disc_cost is the difference between the mean of discriminator output for fake a and real
            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            #The generator wants the discriminator to assign high values to fake_data,
            #such that gen_cost is as small as possible (i.e. highly negative)
            #The discriminator wants to assign small values to fake_data and high values to real_data
            #such that they have the largest possible difference and the disc_cost becomes small
            #This defines the adversarial relationship
            
            with tf.name_scope('D_gp'):
                alpha = tf.random_uniform(shape=[BATCH_SIZE/len(DEVICES),1],minval=0.,maxval=1.)
                #alpha is a tensor of size [BATCH_SIZE/N_GPUS, 1] with values in [0,1]
                #alpha = 0 -> interpolate = real_data_sample
                #alpha = 1 -> interpolate = fake_data_sample
                #in-between values gives intermediate points

                #REAL_DATA = BATCH_SIZE / N_GPUS , SPECTRAL-RESOLUTION; WIDTH, height
                flat_shape = [BATCH_SIZE/len(DEVICES), OUTPUT_DIM]
                real_data_flat, fake_data_flat = tf.reshape(real_data, flat_shape), tf.reshape(fake_data, flat_shape)
                                
                interpolates = (1.0-alpha)*real_data_flat + (0.0+alpha)*fake_data_flat
                #Interpolates is passed to Discriminator as flattened shape
                #This works as long as Discriminator has a superfluous initial reshape
                gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                #This is computationally demanding - uses backpropagation

                if MODE == 'wgan-lp':
                    #root mean square over spectral/spatial axes
                    gradient_rms = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
                    #mean over batch axis
                    gradient_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0, gradient_rms-1.0)))
                    #square of gradient_mean - 1 : if this result is negative, set to zero
                    disc_cost_gp = disc_cost + LAMBDA*gradient_penalty

                else:
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    #Root of sum of square over the non-batch axis (images are already flattened)
                    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                    disc_cost_gp = disc_cost + LAMBDA*gradient_penalty
            
            #Stores results for each GPU-parallel
            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)
            disc_costs_gp.append(disc_cost_gp)
            disc_reals.append(disc_real)
            fake_datas.append(fake_data)

    #Outputs from work loop - trivial operations to aggregate results from GPU-parallels
    #Single numbers:
    #gen_cost - Generator cost, negative mean of Discriminator applied on generated data
    #disc_cost - Discriminator cost, mean difference of Discriminator(real_data) and Discriminator(fake_data)
    #disc_cost_gp - As disc_cost, but with gradient penalty applied
    #Array of length BATCH_SIZE
    #disc_real - Discriminator scores (sample-wise) for real data
    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)
    disc_cost_gp = tf.add_n(disc_costs_gp) / len(DEVICES)    
    disc_real = tf.concat(disc_reals, 0)
    fake_data = tf.concat(fake_datas, 0)

    #Training ops for generator and discriminator
    gen_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        gen_cost, global_step=global_step, var_list=tf.trainable_variables(scope='G'), colocate_gradients_with_ops=True)

    disc_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_cost_gp,
                                       var_list=tf.trainable_variables(scope='D'), colocate_gradients_with_ops=True)
    
    #Generate image from noise - noise as variable allows optimization with regards to noise    
    fixed_noise = np.random.normal(size=[BATCH_SIZE, NOISE_DIM]).astype(np.float32)
    noise = tf.Variable(tf.zeros([BATCH_SIZE, NOISE_DIM], dtype=tf.float32), name='gen_in')
    
    gen_from_noise_collector = []
    with tf.name_scope('G_draw'):
        for device_index, device in enumerate(DEVICES):
            with tf.device(device), tf.name_scope('device_index'):
                n_samples = BATCH_SIZE / len(DEVICES)
                gen_from_noise_collector.append(
                    Generator(n_samples, noise=noise[device_index*n_samples:(device_index+1)*n_samples])
                    )
    gen_from_noise = tf.concat(gen_from_noise_collector, axis=0, name='gen_out')

    #Batch minimize stuff
    init_op_noise = tf.assign(noise, tf.random_normal(shape=[BATCH_SIZE, NOISE_DIM]))
    #Define cost function - returns sample-wise cost
    batch_reconstruction_cost = batch_cost(batch_real_data, gen_from_noise, noise)
    
    #Provides minimizer op and initializer for finding best noise such that cost(noise) is minimal
    with tf.variable_scope('Optimizer'):
        #minimizer = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1e-4)
        minimizer = tf.train.MomentumOptimizer(learning_rate=1e-1, momentum = 0.9)
        minimize_op = minimizer.minimize(tf.reduce_sum(batch_reconstruction_cost), var_list = [noise])
    init_op_minimizer = tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Optimizer'))

    def generate_image(iteration):
        samples = session.run(gen_from_noise, feed_dict = {noise : fixed_noise})
        lib.save_images.save_images(samples, 'generator_image_sample_{}.png'.format(iteration))
        #Generates image from noise that remains unchanged for a given run
        #Saves these to disk - function is called in EVAL in train loop

    #Generators that output images from dataset
    #Modified to use Bjorkelangen-loader
    train_gen, validation_gen, anomaly_gen = lib.ffi_bjrkl.load(BATCH_SIZE, data_dir=DATA_DIR)

    #Normal generators are limited to a single epoch - this constructs an infinite generator for training
    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

    #Saves a BATCH_SIZE of real data samples from training data as GROUNDTRUTH
    #Directly comparable with generated images
    lib.save_images.save_images(inf_train_gen().next(), 'generator_image_groundtruth.png')
   
    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    #saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G'))
    gen = inf_train_gen()
    
    
    if MODEL_PATH != '':
        #Skip training and load model directly
        saver.restore(session, MODEL_PATH)
        #Setting ITERS to -1 skips the training for-loop
        #Avoids another layer of nesting/indentation
        ITERS = -1
        #Removing this assign will allow resuming training of a loaded model
        #Note that generators, iteration count and logs will be reset

    #Train loop
    for iteration in xrange(ITERS+1):
        start_time = time.time()
        #Train generator
        _ = session.run(gen_train_op)
        
        #Train discriminator - CRITIC_ITERS loops per generator loop
        for i in xrange(CRITIC_ITERS):
            _disc_cost, _disc_cost_gp, _ = session.run([disc_cost, disc_cost_gp, disc_train_op], feed_dict={batch_real_data: gen.next()})
        
        #These values are dumped -every- iteration
        lib.plot.plot('discriminator cost:std,gp', [_disc_cost, _disc_cost_gp])
        lib.plot.plot('iteration time', time.time() - start_time)

        eval_time = time.time()

        #When we run logging, every EVAL_INTERVAL intervals, calculate metrics and send to lib.plot.plot:
        if iteration != 0 and (iteration % EVAL_INTERVAL == 0):
            t = time.time()

            #Test the discriminator on EVAL_BATCHES number of batches from validation data
            disc_cost_val_samplewise = np.concatenate(
                [session.run(disc_real, feed_dict={batch_real_data: images}) for (images,) in islice(validation_gen(), EVAL_BATCHES)])
            #Test the discriminator non-adversarially (generator not involved)
            #on real, anomalous images set aside
            disc_cost_anom_samplewise = np.concatenate(
                [session.run(disc_real, feed_dict={batch_real_data: images}) for (images,) in islice(anomaly_gen(), EVAL_BATCHES)])
            #Ideally, the discrimnator maps validation images to higher values and anomalous images to lower values
            #Test using discriminator outputs to differentiate anomalies from background
            v_disc = np.mean(disc_cost_val_samplewise)
            a_disc = np.mean(disc_cost_anom_samplewise)
            lib.plot.plot('discriminator score:val,anom,diff', [v_disc,a_disc, a_disc-v_disc] )
            
            #Save ROC plot - Discrimination between anomaly and validation
            #Note minus signs, because expected behavior is high scores for anomalies and low scores for validation
            lib.plot.plot_roc(-disc_cost_val_samplewise, -disc_cost_anom_samplewise)
            #Save Histogram - hist_disc negates and zero-shifts the scores
            lib.plot.hist_disc(disc_cost_val_samplewise, disc_cost_anom_samplewise)

            #Simple accuracy metrics
            #conf = [TP,FP,TN,FN]
            conf = anomaly_confusion_matrix(disc_cost_val_samplewise, disc_cost_anom_samplewise)
            anom_recall = conf[0] / np.sum([conf[0]+conf[3]])
            anom_precision = conf[0] / np.sum([conf[0] + conf[1]])
            anom_f1 = 2*anom_recall*anom_precision/(anom_recall + anom_precision)
            lib.plot.plot('discriminator detection:recall,precision,F1', [anom_recall, anom_precision, anom_f1])
            
            
            #Test generator based reconstruction cost for anomaly discrimination
            reconstruction_cost_val_samplewise =    np.concatenate([batch_minimize(images)[0] for (images,) in islice(validation_gen(), EVAL_BATCHES)])
            reconstruction_cost_anom_samplewise =   np.concatenate([batch_minimize(images)[0] for (images,) in islice(anomaly_gen(),    EVAL_BATCHES)])
            v_gen,a_gen = np.mean(reconstruction_cost_val_samplewise), np.mean(reconstruction_cost_anom_samplewise)
            lib.plot.plot('generator reconstruction cost:val,anom,diff', [v_gen,a_gen, a_gen-v_gen])
            lib.plot.plot_roc_r(reconstruction_cost_val_samplewise, reconstruction_cost_anom_samplewise)
            lib.plot.hist_gen(reconstruction_cost_val_samplewise, reconstruction_cost_anom_samplewise)
            
            #Save sample images generated by Generator
            generate_image(iteration)

            lib.plot.plot('learning rate', session.run(learning_rate))
            #Save model to disk
            _ = saver.save(session, "./model_" + str(iteration))
            lib.plot.plot('evaluation time', time.time() - eval_time)

        #Save plots to disk and print metrics to console, if very early iterations or at even intervals
        if (iteration < 3) or (iteration != 0 and iteration % EVAL_INTERVAL == 0):
          lib.plot.flush()

        lib.plot.tick()

    #After finished training
    end_validation()
    print 'ALL DONE'