import os, sys
sys.path.append(os.getcwd())

import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.save_images
import tflib.ffi_bjrkl
import tflib.plot

import matplotlib.pyplot as plt

from scipy import stats

MINIMIZE_RUNS = 4           # Uses the best results from across this amount of runs
                            # Helps avoid convergence / local minima issues
                            # Can be set to 1
MINIMIZE_ITERATIONS = 500   # Minimization iterations for each run - gradient descent minimization of cost with respect to noise
NOISE_WEIGHT = 1e-3         # Scales the relative importance of pixel-wise similarity
                            # and noise-likelihood
NUMS = [0, 256+1, 256+1]    # Limits number of validation / anomaly samples generated for complete run
                            # Debug tool - allows testing full code in reasonable time
LOAD_PATH = '/home/akeid/ffi-gan/out/tiny-20chn-nonorm-reconstruction-correctkernel/model_2000'
DATA_DIR = '/home/akeid/ffi-gan/data/tiny'

#The following values must match with Generator training
NOISE_DIM = 128 # Defaults to 128 in original implementation - used to be hard-coded
N_GPUS = 4 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
CHANNELS = 20 # Number of spectral channels in dataset
    #Avoiding hard-coding allows us to work with either 20- or 80-chn source data
WINDOW = 5 # Width = height of windows
OUTPUT_DIM = WINDOW*WINDOW*CHANNELS # Number of pixels in each iamge
    #This must match our image dimensionality - window_size and number_of_channels
    #Use declared constant instead of magic constant for channels
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)] #GPU-naming

#Cost function
#Uses L2-norm of pixelwise difference between real and fake images
#Uses log-likelihood of noise
#Sums each cost contribution with NOISE_WEIGHT for noise
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
            #For the case where _real and _fake are independent, i.e. the Generator is clueless

            batch_diff = tf.subtract(_real, _fake)
            batch_diff_norm = tf.reduce_sum(tf.square(batch_diff), axis=[1,2,3]) / OUTPUT_DIM
            
            batch_noise_norm = tf.reduce_sum(tf.square(_noise), axis=[1]) / NOISE_DIM
            cost_collector.append(batch_diff_norm + NOISE_WEIGHT * batch_noise_norm)

    #Return collected results concat'ed into shape=[BATCH_SIZE]
    return tf.concat(cost_collector, 0)

def batch_minimize_log(batch_real, batch_name='empty', log=False):
    #Initialize gen_input to random noise 
        
    #Collects batch scores across runs
    batch_score_collector = []

    #save real images
    if log: lib.save_images.save_images(batch_real, batch_name + '.png')
    
    #For each initial noise vector tested
    for min_run in xrange(MINIMIZE_RUNS):
        _,_ = session.run([init_op_gen_input, init_op_minimizer])
        
        name_run_postfix = ''
        if MINIMIZE_RUNS > 1: name_run_postfix = '_' + str(min_run) 
        batch_name_run = batch_name + name_run_postfix

        for min_iter in xrange(MINIMIZE_ITERATIONS):
            _, batch_scores_run, fake_data = session.run([minimize_op, gen_cost, gen_output], feed_dict = {real_data : batch_real})
            if log:
                batch_stats = stats.describe(batch_scores_run)
                #stats.describe indices: 1: mean cost, 2: max and min as tuple, 3: standard deviation of cost
                lib.plot.plot(batch_name_run + ' cost', batch_stats[2])
                lib.plot.plot(batch_name_run + ' cost spread', batch_stats[1][1] - batch_stats[1][0])
                lib.plot.plot(batch_name_run + ' cost std', np.sqrt(batch_stats[3]))
                if min_iter % (MINIMIZE_ITERATIONS / 3) == 0:
                    lib.save_images.save_images(fake_data, batch_name_run + '_like_{}.png'.format(min_iter))
                    fake_data_sort_by_cost = fake_data[batch_scores_run.argsort()]
                    lib.save_images.save_images(fake_data_sort_by_cost, batch_name_run + '_like_{}'.format(min_iter) + '_increasing_cost.png')
                    lib.plot.flush()
                lib.plot.tick()

        if log: lib.plot.tick_reset()
        batch_score_collector.append(batch_scores_run)

    batch_scores = np.amin(batch_score_collector, axis=0)
    return batch_scores

def batch_minimize(batch_real, batch_name=None):
    #Do minimzation a number of times with different initial values
    if batch_name != None: lib.save_images.save_images(batch_real, batch_name + '.png')
    
    for min_run in xrange(MINIMIZE_RUNS):
        #Initialize gen_input with random noise - reset optimizer
        _,_ = session.run([init_op_gen_input, init_op_minimizer])
        #Minimize noise for a number of iterations
        for min_iter in xrange(MINIMIZE_ITERATIONS):
            _, batch_scores_run, batch_fake_run = session.run([minimize_op, gen_cost, gen_output], feed_dict = {real_data : batch_real})
        #Keep lowest score and corresponding image for each
        #If first run - nothing to compare with
        if min_run == 0:
            batch_scores_min = batch_scores_run
            batch_fake_min = batch_fake_run
        else:
            m = np.argmin([batch_scores_min, batch_scores_run], axis=0)
            for i in xrange(BATCH_SIZE):
                if m[i]:
                    batch_scores_min[i] = batch_scores_run[i]
                    batch_fake_min[i] = batch_fake_run[i]
        if batch_name != None:
            lib.save_images.save_images(batch_fake_min, batch_name + '_like_{}.png'.format(min_run))

    print 'eob'
    return batch_scores_min, batch_real, batch_fake_min

def anomaly_detection(validation_gen, anomaly_gen):
    #Does cost-minimization on every batch from validation and anomaly generators
    #Plots ROC curves and saves images to folder ./sorteds
    #Each file in this folder contains a real image and the minimal generated image
    #File has leading name fp,tp,fn,tn according to classification
    #Within each type, increasing index corresponds to higher score
    #i.e. higher estimated likelihood of anomaly

    anom = {}
    val = {}
    pack_keys = ['scores', 'real_images', 'fake_images']

    for d,d_gen in zip((val,anom),(validation_gen, anomaly_gen)):
        batch_return = [batch_minimize(images) for (images,) in d_gen()]
        for i in xrange(3):
            d[pack_keys[i]] = np.concatenate([e[i] for e in batch_return], axis=0)
    print '\nBatches done'


    #Pick threshold from sorted scores, such that number of positives and negatives will always be correct
    threshhold = np.sort(np.concatenate([val['scores'], anom['scores']],axis=0))[len(val['scores'])]
    
    indices_ascending_order = np.argsort(val['scores'])
    tn,fp = 0,0
    for i in xrange(len(indices_ascending_order)):
        ind = indices_ascending_order[i]
        im = np.array([ val['real_images'][ind], val['fake_images'][ind] ])
        if val['scores'][ind] >= threshhold:
            lib.save_images.save_images(im, './sorteds/fp_{}.png'.format(str(fp)))
            fp += 1
        else:
            lib.save_images.save_images(im, './sorteds/tn_{}.png'.format(str(tn)))
            tn += 1

    #Difficult to maintain code-duplication --
    indices_ascending_order = np.argsort(anom['scores'])
    tp,fn = 0,0
    for i in xrange(len(indices_ascending_order)):
        ind = indices_ascending_order[i]
        im = np.array([ anom['real_images'][ind], anom['fake_images'][ind] ])
        if anom['scores'][ind] >= threshhold:
            lib.save_images.save_images(im, './sorteds/tp_{}.png'.format(str(tp)))
            tp += 1
        else:
            lib.save_images.save_images(im, './sorteds/fn_{}.png'.format(str(fn)))
            fn += 1

    v,a = stats.describe(val['scores']), stats.describe(anom['scores'])
    print 'Validation stats', v
    print 'Anomaly stats', a
    
    lib.plot.plot_roc_r(-val['scores'], -anom['scores'])
    lib.plot.flush()

    val_hist = val['scores']
    anom_hist = anom['scores']
    #Truncate max anomaly scores to get proper histogram window
    hist_max = 2*np.mean(anom_hist)
    anom_hist[anom_hist>hist_max] = hist_max
    val_hist[val_hist>hist_max] = hist_max


    plt.hist([val_hist, anom_hist], color=['g','r'], label=['Validation', 'Anomaly (truncated)'], alpha=0.8, bins=25)

    plt.title('Histogram of reconstruction scores')
    plt.xlabel('Reconstruction score')
    plt.ylabel('Sample count')

    plt.legend(loc='upper right')
    plt.savefig('Cost histogram')
    plt.close()


#Provide access to image data through generators
#nums must be smaller than number of images in data folder
_, validation_gen, anomaly_gen = lib.ffi_bjrkl.load_pkl(BATCH_SIZE, data_dir=DATA_DIR, nums=NUMS)

def get_batch(generator):
    def inf_gen(generator):
        while True:
            for (images,) in generator():
                yield images
    return inf_gen(generator).next()

validation_batch = get_batch(validation_gen)
anomaly_batch = get_batch(anomaly_gen)

#Gets a whole batch of copies of single element : debug tool
validation_batch_clones = np.tile(validation_batch[0], [BATCH_SIZE,1,1,1])
anomaly_batch_clones = np.tile(anomaly_batch[0], [BATCH_SIZE,1,1,1])

def adam_variables_initializer(adam_opt, var_list):
    adam_vars = [adam_opt.get_slot(var, name)
                 for name in adam_opt.get_slot_names()
                 for var in var_list if var is not None]
    adam_vars.extend(list(adam_opt._get_beta_accumulators()))
    return tf.variables_initializer(adam_vars)

with tf.Session() as session:
    #Path for trained generator
    #Load trained generator (graph only)
    new_saver = tf.train.import_meta_graph(LOAD_PATH + '.meta')

    graph = tf.get_default_graph()
    gen_output = graph.get_tensor_by_name('gen_out:0')
    gen_input = [v for v in tf.global_variables() if v.name == "gen_in:0"][0]
    real_data = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, CHANNELS, WINDOW, WINDOW])

    #We can now do session.run(gen_output)
    #To get generator outputs from noise vector gen_input
    #gen_input  : [BATCH_SIZE, NOISE_DIM] = [64, 128]
    #gen_output : [BATCH_SIZE, CHANNELS, WINDOW, WINDOW] = [64,16,32,32]
 
    #Initialize op: set gen_input to random noise
    init_op_gen_input = tf.assign(gen_input, tf.random_normal(shape=[BATCH_SIZE, NOISE_DIM]))
    #Define cost function - returns sample-wise cost
    gen_cost = batch_cost(real_data, gen_output, gen_input)

    #tf.reduce_sum aggregates cost for entire 
    with tf.variable_scope('OptimizerLocal'):
        minimizer = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-6)
        minimize_op = minimizer.minimize(tf.reduce_sum(gen_cost), var_list = [gen_input])

    init_op_minimizer = tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'OptimizerLocal'))

    #Load trained generator weights (must be after initialize)
    new_saver.restore(session, LOAD_PATH)

    
    print 'Start debug runs'

    '''
    feature_batch_container = np.ones(shape = [128,64,128], dtype=np.float32)
    linspacer = np.linspace(-5,5,64, dtype=np.float32)
    for i in xrange(128): feature_batch_container[i,:,i] = linspacer

    for i in xrange(128):
        fake_data = session.run(gen_output, feed_dict = {gen_input : feature_batch_container[i,:,:]})
        lib.save_images.save_images(fake_data, 'fake' + '_feature_{}.png'.format(i))
    
    

    #batch_minimize_log(validation_batch_clones, 'clone_validation', log=True)
    #batch_minimize_log(anomaly_batch_clones, 'clone_anomaly', log=True)
    #batch_minimize_log(validation_batch, 'validation', log=True)
    #batch_minimize_log(anomaly_batch, 'anomaly', log=True)
    

    for i in range(4):
        _ = session.run(init_op_gen_input)
        fake_data = session.run(gen_output)
        fake_image = fake_data[0]
        fake_image_as_batch = np.tile(fake_image, [64,1,1,1])

        batch_minimize(fake_image_as_batch, 'fake {} clones'.format(str(i)))
    
    print 'Finished debug runs - start full runs'
    '''
   
    print 'BATCHES: ', (NUMS[1]+NUMS[2]-2)/BATCH_SIZE
    print 'MIN_ITERS: ', MINIMIZE_ITERATIONS
    print 'MIN_RUNS: ', MINIMIZE_RUNS


    anomaly_detection(validation_gen, anomaly_gen)    
      
    print 'Finished full runs'