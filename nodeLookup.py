from __future__ import absolute_import, division, print_function

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


FLAGS = tf.compat.v1.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.



class NodeLookup():
  """Converts integer node ID's to human readable labels."""
  label_lookup_path=""
  uid_lookup_path=""
  node_lookup = None
  graph_def = ""



  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):

    #print("juhuu " + FLAGS.model_dir)
    if not label_lookup_path:
      self.label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      self.uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')

    self.node_lookup = self.load()



  def load(self):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """



    #print("juhuu4 " + self.uid_lookup_path + "   " + self.label_lookup_path)
    if not tf.gfile.Exists(self.uid_lookup_path):
        print("uid_lookup_path not exist")
        tf.logging.fatal('File does not exist %s', self.uid_lookup_path)
    if not tf.gfile.Exists(self.label_lookup_path):
        print("label_lookup_path not exist")
        tf.logging.fatal('File does not exist %s', self.label_lookup_path)
    #print("juhuu4 " + self.uid_lookup_path)
    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(self.uid_lookup_path).readlines()

    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
        parsed_items = p.findall(line)
        uid = parsed_items[0]
        human_string = parsed_items[2]
        uid_to_human[uid] = human_string
        #print(uid + " ::: " + uid_to_human[uid] + " ::: " + human_string)



    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(self.label_lookup_path).readlines()
    for line in proto_as_ascii:
        if line.startswith('  target_class:'):
            target_class = int(line.split(': ')[1])
        if line.startswith('  target_class_string:'):
            target_class_string = line.split(': ')[1]
            node_id_to_uid[target_class] = target_class_string[1:-2]
            #print(str(target_class) + " " + node_id_to_uid[target_class])

    #print ("Length : %d" % len(node_id_to_uid))
    # Loads the final mapping of integer node ID to human-readable string

    #for i in node_id_to_uid:
     #   print("aaaa "+ str(i) + node_id_to_uid[i] )


    node_id_to_name = {}
    #print("juhuu4" )

    for i in node_id_to_uid:
        key=i
        val=node_id_to_uid[i]
        #print(val)
        #print(uid_to_human[val])
        if val not in uid_to_human:
            print("feil under oppslag")
            tf.logging.fatal('Failed to locate: %s', val)
        #print("juhuu5")
        #print(uid_to_human[val])
        name = uid_to_human[val]
        #print(name)
        node_id_to_name[key] = name

    #for i in node_id_to_name:
         #print("aaaa "+ str(i) + node_id_to_name[i] )

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]