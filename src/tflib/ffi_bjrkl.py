import numpy as np
import scipy.misc
import pickle
import time




#Repurposed code for use with 16-channel bjrkl-data
def make_generator_pkl(path, n_files, batch_size):
    epoch_count = [1]
    def get_epoch():
        #Hardcoded image size 20*5*5
        images = np.zeros((batch_size, 20, 5, 5), dtype='float32')
        files = range(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = pickle.load(open(path+'/'+str(i)+'.pkl', 'rb'))
            images[n % batch_size] = image

            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

#Hardcoded number of images in data folders
def load_pkl(batch_size, data_dir=None, nums=[221148, 55286, 285]):
    #nums : number of samples contained in each folder
    #just to avoid having to figure this out during runtime
    return (
        make_generator_pkl(data_dir+'/train', nums[0], batch_size),
        make_generator_pkl(data_dir+'/validation', nums[1], batch_size),
        make_generator_pkl(data_dir+'/anomaly', nums[2], batch_size)
    )

def make_generator(path, batch_size):
    epoch_count = [1]
    def get_epoch():
        all_images = pickle.load(open(path, 'rb'))
        batch_images = np.zeros((batch_size,)+all_images[0].shape, dtype='float32')
        
        image_count = all_images.shape[0]
        image_indices = range(image_count)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(image_indices)
        epoch_count[0] += 1
        for i in xrange(image_count):
            batch_images[i % batch_size] = all_images[image_indices[i]]
            if i > 0 and i % batch_size == 0:
                yield (batch_images,)
    return get_epoch

def load(batch_size, data_dir=None):
    #Loads from single .pkl file
    return (
        make_generator(data_dir+'/train.pkl', batch_size),
        make_generator(data_dir+'/val.pkl', batch_size),
        make_generator(data_dir+'/anom.pkl', batch_size)
    )




if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()