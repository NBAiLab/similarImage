from nltk import ngrams
import random, json, glob, os, codecs, random
import numpy as np
import os.path
import re
import sys
import tarfile
import glob
import json
import psutil
from collections import defaultdict
import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")



def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def makeImageVector(image, output_dir):

    #create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        if not tf.gfile.Exists(image):
            tf.logging.fatal('File does not exist %s', image)

        with tf.gfile.FastGFile(image, 'rb') as f:
            image_data = f.read()

            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})

            predictions = np.squeeze(predictions)

            ###
            # Get penultimate layer weights
            ###

            feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            feature_set = sess.run(feature_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            feature_vector = np.squeeze(feature_set)
            outfile_name = os.path.basename(image) + ".npz"
            out_path = os.path.join(output_dir, outfile_name)
            print(feature_vector)
            np.savetxt(out_path, feature_vector, delimiter=',')

            # Creates node ID --> English string lookup.



    return "image_to_labels"

def main(_):

  output_dir="yy"

  create_graph()
  image_to_labels = makeImageVector(sys.argv[1], output_dir)
  print("all done")


if __name__ == '__main__':
  tf.app.run()
