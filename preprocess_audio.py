from tqdm import tqdm
import tensorflow as tf
from models import vggish_slim
from utils import vggish_input
from utils import vggish_params
from utils import vggish_postprocess
import numpy as np
import pandas as pd
import os


def get_embeddings(path, sess):
    tf.reset_default_graph()
    try:
        examples_batch = vggish_input.wavfile_to_examples(path)
        [embedding_batch] = sess.run([embedding_tensor],
                         feed_dict={features_tensor: examples_batch})

        if(movie['YT-Trailer ID'] == movie['YT-Trailer ID']):
            return(np.mean(embedding_batch, axis = 0))
        else:
            print("No")
    except:
        pass

sess = tf.Session()
vggish_slim.define_vggish_slim(training=False)
vggish_slim.load_vggish_slim_checkpoint(sess, "./pretrained/vggish_model.ckpt")
features_tensor = sess.graph.get_tensor_by_name(
    vggish_params.INPUT_TENSOR_NAME)
embedding_tensor = sess.graph.get_tensor_by_name(
    vggish_params.OUTPUT_TENSOR_NAME)

for index, movie in items.iterrows():
    path = "/scratch/zz4330/ml-100k/Audio/{}.wav".format(movie['YT-Trailer ID'])
    if(movie['YT-Trailer ID'] in embeddings.columns):
        continue
    elif(os.path.exists(path) == False):
        continue
    elif(os.path.getsize(path)>=100000000):
        continue
    else:
        print(os.path.getsize(path))
        embeddings[movie['YT-Trailer ID']] = get_embeddings(path, sess)
    if(index % 10 == 0):
        print("Updating")
        pd.DataFrame(embeddings).to_csv("/scratch/zz4330/ml-100k/Audio/embeddings2.csv")

pd.DataFrame(embeddings).to_csv("/scratch/zz4330/ml-100k/Audio/embeddings2.csv")
