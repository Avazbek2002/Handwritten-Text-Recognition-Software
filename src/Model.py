import os

import numpy as np
import tensorflow as tf

# Disable eager mode
tf.compat.v1.disable_eager_execution()



class Model:
    "minimalistic TF model for HTR"

    # model constants
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, mustRestore=False, dump=False):
        "init model: add CNN, RNN and CTC and initialize TF"
        self.dump = dump
        self.charList = charList
        self.mustRestore = mustRestore
        self.snapID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.compat.v1.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

    def toSparse(self, texts):

        encoded_words = []
        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            encoded_words.append([labelStr])

        label = np.zeros(shape = (len(texts), 32, 80))
        for i in range(len(texts)):
            for j in range(len(texts[i])):
                label[i][j][encoded_words[i][j]] = 1

        return label


    def decoderOutputToText(self, ctcOutput, batchSize):
        "extract texts from output of CTC decoder"

        # TF decoders: label strings are contained in sparse tensor

        # ctc returns tuple, first element is SparseTensor
        decoded = ctcOutput[0][0]

        # contains string of labels for each batch element
        labelStrs = [[] for _ in range(batchSize)]

        # go over all indices and save mapping: batch -> values
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batchElement = idx2d[0]  # index according to [b,t]
            labelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in labelStrs]

    def dumpNNOutput(self, rnnOutput):
        "dump the output of the NN to CSV file(s)"
        dumpDir = '../dump/'
        if not os.path.isdir(dumpDir):
            os.mkdir(dumpDir)

        # iterate over all batch elements and create a CSV file for each one
        maxT, maxB, maxC = rnnOutput.shape
        for b in range(maxB):
            csv = ''
            for t in range(maxT):
                for c in range(maxC):
                    csv += str(rnnOutput[t, b, c]) + ';'
                csv += '\n'
            fn = dumpDir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

