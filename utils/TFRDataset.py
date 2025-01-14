import numpy as np
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

def read_tfrecord(example, timesteps=64, height=32, width=32):
    features = {
        'label': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = example['image']
    image = tf.io.decode_raw(image, tf.uint8)
    # image = tf.reshape(image, [64, 128, 128, 3])
    # Downsized dataset
    image = tf.reshape(image, [timesteps, height, width, 3])
    # image = tf.reshape(image, [128, 16, 16, 3])
    # image = tf.reshape(image, [64, 128, 128, 3])

    label  = example['label']
    # height = example['height']
    # width  = example['width']

    return image, label #, height, width
    

def tfr_data_loader(data_dir="", batch_size=32, drop_remainder=True, shuffle_buffer=1000, timesteps=64, height=32, width=32):
    '''
    Function that takes path to tfrecord files (allows regular expressions), 
    and returns a tensorflow dataset that can be iterated upon, 
    using loops or enumerate()
    '''

    if data_dir is None:
        raise ValueError("Missing path to data directory!")
    else:
        data_dir=data_dir
    
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.io.gfile.glob(data_dir)
    dataset = tf.data.TFRecordDataset(dataset, compression_type='GZIP') # , cycle_length=batch_size, num_parallel_calls=8)
    dataset = dataset.map(lambda x: read_tfrecord(x, height=height, width=width, timesteps=timesteps), num_parallel_calls=AUTO)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset

# start=time.time()
# for x in a:
#     tt=torch.from_numpy(x[0].numpy())
#     tt=tt.permute(0,4,1,2,3)
#     print(tt.shape)
# print(time.time()-start)
# np.array(list(map(ord, p[1][1].numpy())))
# np.vectorize(ord)(p[1][1].numpy())
