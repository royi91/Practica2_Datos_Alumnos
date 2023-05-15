# @brief LdaNormalBayesClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2023
import os

# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminarla

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.lda = None
        self.classifier = None

    def train(self, images_dict):
        """.
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """

        # Take training images and do feature extraction
        
        X = [] # Feature vectors by rows
        y = [] # Labels for each row in X

        for dir in os.listdir(images_dict):
            if dir == "may" or dir == "min":
                m = 0
                #for dir2 in os.listdir(images_dict+"\\" + dir):
                    #for img in os.listdir(images_dict+"\\" + dir + "\\" + dir2):
                        #cv2.adaptiveThreshold(img, img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        #cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        #cv2.boundingRect(img)
                        #X.append(img)
                        #y.append(self.char2label(dir2))
            else:
                m = 0
                for img in os.listdir(images_dict+"\\" + dir):
                    img2 = cv2.imread(images_dict+"\\"+dir+"\\"+img)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    img2 = cv2.resize(img2, self.ocr_char_size)
                    img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    img2 = img2.astype(np.float32)
                    #img2 = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # if m == 0:
                    #     cv2.imshow("img", img2)
                    #     cv2.waitKey(0)
                    #     m=1
                    X.append(img2.reshape(-1,self.ocr_char_size[0]*self.ocr_char_size[1]))
                    print("Etiqueta: " + self.label2char(self.char2label(dir)))
                    y.append(self.char2label(dir))
        
        # Perform LDA training
        self.lda = LinearDiscriminantAnalysis()
        X = np.array(X)
        X = X.reshape(-1, self.ocr_char_size[0]*self.ocr_char_size[1])

        self.lda.fit(X, y)


        # Perform Classifier training
        self.classifier = cv2.ml.NormalBayesClassifier_create()
        print("1")
        samples = np.array(X)
        labels = np.array(y)
        self.classifier.train(samples, cv2.ml.ROW_SAMPLE, labels)
        print("2")

        return samples, labels

    def predict(self, img):
        """.
        Given a single image of a character already cropped classify it.

        :img Image to classify
        
        """

        y = ...  # Obtain the estimated label by the LDA + Bayes classifier
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.ocr_char_size)
        img= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = img. astype(np.float32)
        img = img.reshape(1, -1) # Reshape a un solo ejemplo

        _, result = self.classifier.predict(img)
        label = result[0, 0]
        char = self.label2char(label)
        return char



