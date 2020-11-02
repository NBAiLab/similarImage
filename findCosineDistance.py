import sys
import os
import xmlHandler
import elasticsearchHandler
import argparse
import json
import glob


from nodeLookup import NodeLookup
from similarSearchBuilder import SimilarSearchBuilder



if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('inputFile1', help='absolute path to jpg image 1')
        parser.add_argument('inputFile2', help='absolute path to jpg image 2')


        args = parser.parse_args()
        s = SimilarSearchBuilder()
        print("Similarity: " + str(s.compareImages(args.inputFile1,args.inputFile2)))