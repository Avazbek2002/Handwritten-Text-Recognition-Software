import random
import cv2
import numpy as np

from SamplePreprocessor import preprocess


class Sample:
    "sample from the dataset"

    def __init__(self, gtText, preprocessed_img):
        self.gtText = gtText
        self.preprocessed_img = preprocessed_img

class DataLoaderIAM:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, data_dir, imgSize, maxTextLen):
        "loader for dataset at given location, preprocess images and text according to parameters"

        self.imgSize = imgSize
        self.samples = []

        f = open(data_dir + '/gt/words.txt')
        chars = set()
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
        count = 0
        for line in f:
            # ignore comment line
            count = count + 1
            if count == 101:
                break
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')

            fileName = data_dir + '/img/' + fileNameSplit[0] +"/" + f'{fileNameSplit[0]}-{fileNameSplit[1]}' + "/" + lineSplit[0] + '.png'

            img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

            # preprocessed image is (128, 32, 1) between [-1, 1]
            img_processed = preprocess(img, (128, 32))

            # ignores the corrupt images
            if lineSplit[0] in bad_samples_reference:
                print('Ignoring known broken image:', fileName)
                continue

            # GT text are columns starting at 9
            gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
            chars = chars.union(set(list(gtText)))

            # put sample into list
            self.samples.append(Sample(gtText, img_processed))

        # put words into lists (ground truth)
        self.words = [x.gtText for x in self.samples]

        # start with train set
        #self.Set()

        # list of all chars in dataset. Overall charList contains 79 characters (it doesn't include blank symbol)
        self.charList = sorted(list(chars))
        self.charList.insert(0, " ")



    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text


    # converts ground truth text in string ecndes using one-hot encoding
    def toSparse(self, texts):
        # texts contains list of label words in strings
        texts = np.array(texts)
        encoded_words = []
        # go over all texts
        for (index, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text] # ['c', 'a', 't']
            # stores the words' position in charList list e.g. ["cat"] is stored as [2, 0, 20]
            # []
            encoded_words.append(labelStr)

        label = np.zeros(shape = (len(texts), 32))

        for i in range(len(texts)):
            for j in range(len(texts[i])):
                label[i][j] = encoded_words[i][j]

        return label