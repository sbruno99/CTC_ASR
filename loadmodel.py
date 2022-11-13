import keras
import tensorflow as tf
from ctcModel import CTCLoss

import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
import numpy as np


testfile = "TIMIT/TEST/DR1/FAKS0/SA1.WAV"



def features_from_wav(wavfile):
    wavfile = wav.read(wavfile)
    coefs = mfcc(signal=wavfile[1], appendEnergy=True)
    deltas = delta(coefs, 1)
    features = np.hstack((coefs, deltas))

    return features

#model = tf.keras.models.load_model("model", custom_objects={"CTCLoss": CTCLoss})

model = keras.models.load_model("ctc_model.h5")

prediction = model.predict(features_from_wav(testfile))