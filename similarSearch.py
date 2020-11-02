from annoy import AnnoyIndex
from scipy import spatial
from nltk import ngrams
import random, json, glob, os, codecs, random
import numpy as np
import psutil
from collections import defaultdict
from nodeLookup import NodeLookup
from six.moves import urllib
import tensorflow as tf

from operator import itemgetter, attrgetter, methodcaller

# data structures
file_index_to_file_name = {}
file_index_to_file_vector = {}
chart_image_positions = {}

# config
dims = 2048
n_nearest_neighbors = 30
trees = 1000
#infiles = glob.glob('image_vectors_tmp/*.npz')

class SimilarSearch():
    FLAGS = tf.app.flags.FLAGS
    vector_dir=""
    json_dir=""
    image_dir=""
    graphCreated = False
    graph_def=""
    tf

    tf.app.flags.DEFINE_string(
        'model_dir', 'model',
        """Path to classify_image_graph_def.pb, """
        """imagenet_synset_to_human_label_map.txt, and """
        """imagenet_2012_challenge_label_map_proto.pbtxt.""")
    tf.app.flags.DEFINE_string('image_file', '',
                               """Absolute path to image file.""")
    tf.app.flags.DEFINE_integer('num_top_predictions', 3,
                                """Display this many predictions.""")

    def __init__(self, master_dir=None):
        if master_dir == None:
            self.vector_dir = "VECTORS"
            self.image_dir = "IMAGES"
            self.json_dir = "JSON"
        else:
            self.vector_dir = master_dir + "/VECTORS"
            self.image_dir = master_dir + "/IMAGES"
            self.json_dir = master_dir + "/JSON"

    def create_graph(self):

        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.GFile(os.path.join(
                self.FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(self.graph_def, name='')
            self.tf=tf

    def searchSimilarImage(self, image):
        image_to_labels = defaultdict(list)
        if self.graphCreated == False:
            self.create_graph()
            self.graphCreated = True

        
        with self.tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
           
            try:
                print("parsing", image, "\n")
                if not tf.gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)

                with tf.gfile.GFile(image, 'rb') as f:
                    image_data = f.read()
                    predictions = sess.run(softmax_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    predictions = np.squeeze(predictions)

                    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                    feature_set = sess.run(feature_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    feature_vector = np.squeeze(feature_set)


                    # Creates node ID --> English string lookup.
                    #node_lookup = NodeLookup()

                    top_k = predictions.argsort()[-self.FLAGS.num_top_predictions:][::-1]
                    #print(top_k)

                    dims = 2048
                    n_nearest_neighbors = 40
                    trees = 1000
                    t = AnnoyIndex(dims)
                    t.add_item(0, feature_vector)

                    file_name = os.path.basename(image).split('.')[0]
                    file_index_to_file_name[0] = file_name
                    file_index_to_file_vector[0] = feature_vector
                    file_index = 1
                    #f=open("searchres.txt","w")
                    for node_id in top_k:
                        candidateDirPattern=self.vector_dir + "/" + str(node_id) + "/*.npz"
                        infiles = glob.glob(candidateDirPattern)

                        for file_idx, i in enumerate(infiles):
                            #print("xxxx" + i)
                            #f.write("xxxx" + i +"\n")
                            file_vector = np.loadtxt(i)
                            file_name = os.path.basename(i).split('.')[0]
                            file_index_to_file_name[file_index] = file_name
                            file_index_to_file_vector[file_index] = file_vector

                            t.add_item(file_index, file_vector)
                            file_index += 1
                    t.build(trees)
                    #f.close()
                    #print("hei")
                    nearest_neighbors = t.get_nns_by_item(0, n_nearest_neighbors)
                    #print("hei")
                    results=[]
                    for j in nearest_neighbors:
                        neighbor_file_name = file_index_to_file_name[j]
                        neighbor_file_vector = file_index_to_file_vector[j]

                        similarity = 1 - spatial.distance.cosine(feature_vector, neighbor_file_vector)
                        rounded_similarity = int((similarity * 10000)) / 10000.0
                        #print("Image: " + image + " has a similar image: " + neighbor_file_name + " Similarity: " + str(rounded_similarity))
                        tuple=(neighbor_file_name,str(rounded_similarity))
                        if j != 0:
                            results.append(tuple)

                    #print(results)
                    results_sorted=sorted(results, key=itemgetter(1))[::-1]
                    #print(results_sorted)
                # close the open file handlers
                proc = psutil.Process()
                open_files = proc.open_files()

                for open_file in open_files:
                    file_handler = getattr(open_file, "fd")
                    os.close(file_handler)
            except:
                print('could not process image index',  'image', image)

        return results_sorted

    def searchSimilarImageDirect(self, image_input_data):

        f = open('imageQueryfile.jpg', 'w+b')
        f.write(image_input_data)
        f.close()
        image='imageQueryfile.jpg'
        image_to_labels = defaultdict(list)
        if self.graphCreated == False:
            self.create_graph()
            self.graphCreated = True

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

            try:
                print("parsing", image, "\n")
                if not tf.gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)

                with tf.gfile.GFile(image, 'rb') as f:
                    image_data = f.read()
                    predictions = sess.run(softmax_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    predictions = np.squeeze(predictions)

                    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                    feature_set = sess.run(feature_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    feature_vector = np.squeeze(feature_set)

                    # Creates node ID --> English string lookup.
                    # node_lookup = NodeLookup()

                    top_k = predictions.argsort()[-self.FLAGS.num_top_predictions:][::-1]
                    # print(top_k)

                    dims = 2048
                    n_nearest_neighbors = 30
                    trees = 1000
                    t = AnnoyIndex(dims)
                    t.add_item(0, feature_vector)
                    file_name = os.path.basename(image).split('.')[0]
                    file_index_to_file_name[0] = file_name
                    file_index_to_file_vector[0] = feature_vector
                    file_index = 1
                    for node_id in top_k:
                        candidateDirPattern = self.vector_dir + "/" + str(node_id) + "/*.npz"
                        infiles = glob.glob(candidateDirPattern)

                        for file_idx, i in enumerate(infiles):
                            # print("xxxx" + i)
                            file_vector = np.loadtxt(i)
                            file_name = os.path.basename(i).split('.')[0]
                            file_index_to_file_name[file_index] = file_name
                            file_index_to_file_vector[file_index] = file_vector
                            t.add_item(file_index, file_vector)
                            file_index += 1
                    t.build(trees)
                    # print("hei")
                    nearest_neighbors = t.get_nns_by_item(0, n_nearest_neighbors)
                    # print("hei")
                    results = []
                    for j in nearest_neighbors:
                        neighbor_file_name = file_index_to_file_name[j]
                        neighbor_file_vector = file_index_to_file_vector[j]

                        similarity = 1 - spatial.distance.cosine(feature_vector, neighbor_file_vector)
                        rounded_similarity = int((similarity * 10000)) / 10000.0
                        # print("Image: " + image + " has a similar image: " + neighbor_file_name + " Similarity: " + str(rounded_similarity))
                        tuple = (neighbor_file_name, str(rounded_similarity))
                        if j != 0:
                            results.append(tuple)

                    # print(results)
                    results_sorted = sorted(results, key=itemgetter(1))[::-1]
                    # print(results_sorted)
                # close the open file handlers
                proc = psutil.Process()
                open_files = proc.open_files()

                for open_file in open_files:
                    file_handler = getattr(open_file, "fd")
                    os.close(file_handler)
            except:
                print('could not process image index', 'image', image)

        return results_sorted