from scipy import spatial
from nltk import ngrams
import random, json, glob, os, codecs, random
import numpy as np
import glob
import argparse
from os import path


file_index_to_file_name = {}
file_index_to_file_vector = {}

#start=100000
def init(vectordir):
    global file_index_to_file_name
    global file_index_to_file_vector
    infiles = glob.glob(vectordir + '/*.npz')
    file_index_to_file_name = {}
    file_index_to_file_vector = {}
    initcnt=0
    for file_index, i in enumerate(infiles):
        file_vector = np.loadtxt(i)
        file_name = os.path.basename(i).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector
        #print("loading: " + i)
        initcnt+=1
        if ((initcnt % 100) == 0):
            print("loaded " + str(initcnt))


def CheckForNeighborFiles(vectordir,neighbordir):
    for filenow in glob.glob(vectordir + '/*.npz'):
        relfilename = os.path.basename(filenow)
        indexName = relfilename.split(".")[0]
        neigborfile= neighbordir + "/" + indexName + ".json"
        doExist=False
        if (path.exists(neigborfile) == True):
            doExist=True
            print(neigborfile + " exists")
        else:
            doExist=False
            print(neigborfile + " does not exist")


def CheckForNeighborFile(vectorfile,neighbordir):

    neigborfile= neighbordir + "/" + vectorfile + ".json"
    doExist=False
    if (path.exists(neigborfile) == True):
        doExist=True
        print(neigborfile + " exists")
    else:
        doExist=False
        print(neigborfile + " does not exist")
    return doExist

def getVectorFromFile(vectordir,pic):
    file_name = os.path.basename(pic).split('.')[0]
    file_vector = np.loadtxt(vectordir + "/" + file_name + ".npz")
    return file_vector

def findClosestToTwoImages(neighbordir,pic1,pic2):
    global file_index_to_file_name
    global file_index_to_file_vector
    cnt = 0


def processimages(neighbordir,start,stop):
    global file_index_to_file_name
    global file_index_to_file_vector
    cnt = 0
    for i in file_index_to_file_name.keys():
        if (cnt >= start and cnt <= stop):
            master_file_name = file_index_to_file_name[i]
            if (CheckForNeighborFile(master_file_name, neighbordir) == False):
                master_vector = file_index_to_file_vector[i]
                named_nearest_neighbors = []
                short_named_nearest_neighbors = []
                cnt+=1

                for j in file_index_to_file_name.keys():
                    neighbor_file_name = file_index_to_file_name[j]
                    neighbor_file_vector = file_index_to_file_vector[j]

                    similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
                    rounded_similarity = int((similarity * 1000000)) / 1000000.0

                    #print(similarity)

                    named_nearest_neighbors.append({
                      'filename': neighbor_file_name,
                      'similarity': similarity})

                named_nearest_neighbors = sorted(named_nearest_neighbors, key=lambda k: k.get('similarity', 0), reverse=True)
                for k2 in range(30):
                    short_named_nearest_neighbors.append(named_nearest_neighbors[k2])

                fileWritten=False
                while fileWritten == False:
                    print("Should build " + neighbordir + "/" + master_file_name + '.json',)
                    try:
                        with open( neighbordir + "/" + master_file_name + '.json', 'w') as out:
                            json.dump(short_named_nearest_neighbors, out)
                            fileWritten = True

                    except Exception as e:
                        print("Exception caught " + e.message)
            else:
                print ("File " + neighbordir + "/" + master_file_name + " already exists")
        else:
            cnt+=1
            print("skipping nr:" + str(cnt) + " outside of count")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('Vectordir', help='dir with vectors')
    parser.add_argument('Neigbordir', help='dir with json neighbor files')
    parser.add_argument('start', help='start image nr')
    parser.add_argument('stop', help='start image nr')

    args = parser.parse_args()
    #CheckForNeighborFiles(args.Vectordir, args.Neigbordir)

    print(args.start + "   " + args.stop)
    init(args.Vectordir)
    print ("Finished init for " + args.start + "and" + args.stop )
    processimages(args.Neigbordir,int(args.start), int(args.stop))