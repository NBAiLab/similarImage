from __future__ import absolute_import, division, print_function

from scipy import spatial
from nltk import ngrams
import random, json, glob, os, codecs, random
import numpy as np
import psutil
from collections import defaultdict
from nodeLookup import NodeLookup
from six.moves import urllib
import tensorflow as tf
import glob
from os import path


class SimilarSearchBuilder():
    FLAGS = tf.compat.v1.app.flags.FLAGS
    graph_def=""
    graphCreated=False

    tf.compat.v1.app.flags.DEFINE_string(
        'model_dir', 'model',
        """Path to classify_image_graph_def.pb, """
        """imagenet_synset_to_human_label_map.txt, and """
        """imagenet_2012_challenge_label_map_proto.pbtxt.""")
    tf.compat.v1.app.flags.DEFINE_string('image_file', '',
                               """Absolute path to image file.""")
    tf.compat.v1.app.flags.DEFINE_integer('num_top_predictions', 10,
                                """Display this many predictions.""")

    def create_graph(self):

        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.compat.v1.gfile.GFile(os.path.join(
                self.FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            self.graph_def = tf.compat.v1.GraphDef()
            self.graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(self.graph_def, name='')

    file_index_to_file_name = {}
    file_index_to_file_vector= {}

    def __init__(self):
        file_index_to_file_vector = {}
        file_index_to_file_name = {}


    def brutForceNearestNeigbour(self,inputdir,outputdir):
        infiles = glob.glob(inputdir + '/*.npz')
        for file_index, i in enumerate(infiles):
            file_vector = np.loadtxt(i)
            file_name = os.path.basename(i).split('.')[0]
            file_index_to_file_name[file_index] = file_name
            file_index_to_file_vector[file_index] = file_vector
            print("loading: " + i)

        for i in file_index_to_file_name.keys():
            master_file_name = file_index_to_file_name[i]
            master_vector = file_index_to_file_vector[i]
            named_nearest_neighbors = []
            short_named_nearest_neighbors = []

            for j in file_index_to_file_name.keys():
                neighbor_file_name = file_index_to_file_name[j]
                neighbor_file_vector = file_index_to_file_vector[j]

                similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
                rounded_similarity = int((similarity * 1000000)) / 1000000.0

                # print(similarity)

                named_nearest_neighbors.append({
                    'filename': neighbor_file_name,
                    'similarity': similarity})

            named_nearest_neighbors = sorted(named_nearest_neighbors, key=lambda k: k.get('similarity', 0),
                                             reverse=True)
            for k2 in range(30):
                short_named_nearest_neighbors.append(named_nearest_neighbors[k2])

            with open(outputdir + '/' + master_file_name + '.json', 'w') as out:
                json.dump(short_named_nearest_neighbors, out)

    def makeImageVector(self, image, output_dir):
        if self.graphCreated == False:
            self.create_graph()
            self.graphCreated = True

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

            if not tf.gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            with tf.gfile.GFile(image, 'rb') as f:
                image_data = f.read()

                #predictions = sess.run(softmax_tensor,
                                      # {'DecodeJpeg/contents:0': image_data})

                #predictions = np.squeeze(predictions)

                ###
                # Get penultimate layer weights
                ###

                feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                feature_set = sess.run(feature_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                feature_vector = np.squeeze(feature_set)
                outfile_name = os.path.basename(image) + ".npz"
                out_path = os.path.join(output_dir, outfile_name)
                #print(feature_vector)
                np.savetxt(out_path, feature_vector, delimiter=',')

                # Creates node ID --> English string lookup.

        return "imageVectorBuildt"

    def makeImageVector(self, image):
        if self.graphCreated == False:
            self.create_graph()
            self.graphCreated = True


        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

            if not tf.gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            f=open(image,"rb")
            image_data = f.read()

            feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            feature_set = sess.run(feature_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            feature_vector = np.squeeze(feature_set)
            return feature_vector


    def compareImages(self,image1,image2):
        V1=self.makeImageVector(image1)
        V2 = self.makeImageVector(image2)

        similarity = 1 - spatial.distance.cosine(V1, V2)
        #rounded_similarity = int((similarity * 1000000)) / 1000000.0
        #print(similarity)
        return similarity

    def makeVectors(self,imagedir,outputdir,ext):
        mappingsString = imagedir + "/" + "*.jpg"
        for filenow in glob.glob(mappingsString):
            if os.path.getsize(filenow) > 500:
                filebasicname=filenow.split("/")[-1].split(".")[0]
                otuputfile=outputdir + "/" + filebasicname + "." + ext
                if not(path.exists(otuputfile)):
                    print("Processing:" + otuputfile)
                    self.makeImageVectorBasic(filenow,otuputfile)
                else:
                    print("File exists:" +  otuputfile)

        
    def makeImageVectorBasic(self, image, outputFile):
        if self.graphCreated == False:
            self.create_graph()
            self.graphCreated = True


        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

            if not tf.compat.v1.gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            with tf.io.gfile.GFile(image, 'rb') as f:
                image_data = f.read()

                #predictions = sess.run(softmax_tensor,
                                      # {'DecodeJpeg/contents:0': image_data})

                #predictions = np.squeeze(predictions)

                ###
                # Get penultimate layer weights
                ###

                feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                feature_set = sess.run(feature_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                feature_vector = np.squeeze(feature_set)
               # outfile_name = os.path.basename(image) + ".npz"
                #out_path = os.path.join(output_dir, outfile_name)
                #print(feature_vector)
                np.savetxt(outputFile, feature_vector, delimiter=',')

                # Creates node ID --> English string lookup.

        return "imageVectorBuildt"


    def processImages(self,input_dir, output_dir):
        """Runs inference on an image list.

        Args:
          input_dir: a dir of images. (I.E. images/*.jpg)
          output_dir: the directory in which image vectors will be saved

        Returns:
          image_to_labels: a dictionary with image file keys and predicted
            text label values
        """
        image_list= glob.glob(input_dir)
        image_to_labels = defaultdict(list)
        if self.graphCreated == False:
            self.create_graph()
            self.graphCreated = True

        with tf.Session() as sess:
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

            for image_index, image in enumerate(image_list):
                try:
                    print("parsing", image_index, image, "\n")
                    if not tf.gfile.Exists(image):
                        tf.logging.fatal('File does not exist %s', image)

                    with tf.gfile.GFile(image, 'rb') as f:
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


                        #out_path = os.path.join(output_dir, outfile_name)
                        #np.savetxt(out_path, feature_vector, delimiter=',')

                        # Creates node ID --> English string lookup.
                        node_lookup = NodeLookup()

                        top_k = predictions.argsort()[-self.FLAGS.num_top_predictions:][::-1]
                        #print("hei4")
                        main_node_set=False
                        for node_id in top_k:
                            if main_node_set == False:
                                main_node_id = str(node_id)
                                main_node_set = True

                            human_string = node_lookup.id_to_string(node_id)
                            score = predictions[node_id]
                            #print("results for", image)
                            #print('%s (score = %.9f)' % (human_string, score))
                            #print(node_id)
                            curr_nodeid=str(node_id)
                            image_to_labels[image].append(
                                {

                                    "labels": human_string,
                                    "score": str(score),
                                    "Category": curr_nodeid
                                }
                            )
                        vector_output_dir = output_dir + "/VECTORS/" + main_node_id
                        if not os.path.exists(vector_output_dir):
                            os.makedirs(vector_output_dir)

                        out_path = os.path.join(vector_output_dir, outfile_name)
                        np.savetxt(out_path, feature_vector, delimiter=',')

                        json_output_dir = output_dir + "/JSON/" + main_node_id
                        if not os.path.exists(json_output_dir):
                            os.makedirs(json_output_dir)

                        with open(json_output_dir + '/' + os.path.basename(image) + '.json', 'w') as out:
                            json.dump(image_to_labels, out)

                        picture_copy_output_dir = output_dir + "/IMAGES/" + main_node_id
                        if not os.path.exists(picture_copy_output_dir):
                            os.makedirs(picture_copy_output_dir)

                        imagefilename=picture_copy_output_dir + "/" + os.path.basename(image)
                        #print(imagefilename)
                        of = open(imagefilename, "wb")
                        of.write(image_data)
                        of.close()
                        #print(imagefilename)
                        image_to_labels = defaultdict(list)
                    # close the open file handlers
                    proc = psutil.Process()
                    open_files = proc.open_files()

                    for open_file in open_files:
                        file_handler = getattr(open_file, "fd")
                        os.close(file_handler)
                except:
                    print('could not process image index', image_index, 'image', image)

        return image_to_labels

    def myprint(self):
        print("hei")

