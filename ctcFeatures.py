from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import numpy as np
import keras

import glob

testfile = "SA1.txt"

TRAINDIR = "TIMIT/TRAIN"
TESTDIR = "TIMIT/TEST"

TRAINOUTCOEFS = "traindatacoefs.npy"
TRAINOUTLABS = "traindatalabs.npy"

TESTOUTCOEFS = "testdatacoefs.npy"
TESTOUTLABS = "testdatalabs.npy"


framesize = 0.0025 # seconds
framebuffer = 0.01  # seconds
samplerate = 16000 # hz
samplesperms = samplerate / 1000
samplesperframe = samplesperms * 25



# accepts a txt file and returns list of char number labels
def makelabels(file):
    chars = "abcdefghijklmnopqrstuvwxyz" # include period?
    charnums = {}
    for i, c in enumerate(chars):
        charnums[c] = i

    f = open(file, "r")
    words = ' '.join(f.readline().split()[2:]).lower()  # eliminate the first two words

    labels = []

    for i, char in enumerate(words):
        if char in charnums:
            labels.append(charnums[char])

    return labels



def makefeatures(wavfile, txtfile):
    wavfile = wav.read(wavfile)
    coefs = mfcc(signal=wavfile[1], appendEnergy=True)
    deltas = delta(coefs, 1)
    # probably inefficient way to combine the rows of the above two vectors
    # to form new nparray
    features = np.hstack((coefs, deltas))

    labels = makelabels(txtfile)
    return features, labels


# for the given input dir create features and
# output to file
def createdata(inputdir, coefsout, labsout):
    print("hi")
    features = []
    labels = []
    pathname = inputdir + "/**/*.wav"
    wavfiles = glob.glob(pathname, recursive=True)
    pathname = inputdir + "/**/*.txt"
    txtfiles = glob.glob(pathname, recursive=True)
    for i in range(0, len(wavfiles)):
        coefs, labs = makefeatures(wavfiles[i], txtfiles[i])
        features.append(coefs)
        labels.append(labs)

    features = keras.utils.pad_sequences(features, padding='post')
    labels = keras.utils.pad_sequences(labels, padding='post')
   # labels = np.stack(list(map(lambda x: keras.utils.to_categorical(x, num_classes=26), labels)), axis=0)
    labels = np.stack(labels)


    np.save(coefsout, features)
    np.save(labsout, labels)


createdata(TRAINDIR, TRAINOUTCOEFS, TRAINOUTLABS)
createdata(TESTDIR, TESTOUTCOEFS, TESTOUTLABS)

