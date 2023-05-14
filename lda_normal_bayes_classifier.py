# @brief LdaNormalBayesClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2023

# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminarla

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .ocr_classifier import OCRClassifier

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
        
        X = ... # Feature vectors by rows
        y = ... # Labels for each row in X 

        for dir in images_dict:
            if dir == "may" or dir == "min":
                for dir2 in images_dict[dir]:
                    for img in images_dict[dir][dir2]:
                        cv2.adaptiveThreshold(img, img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.boundingRect(img)
                        X.append(img)
                        y.append(self.char2label(dir2))
            else:
                for img in images_dict[dir]:
                    cv2.adaptiveThreshold(img, img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.boundingRect(img)
                    X.append(img)
                    y.append(self.char2label(dir))
        
        # Perform LDA training
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X, y)

        # Perform Classifier training
        self.classifier = cv2.ml.NormalBayesClassifier_create()
        self.classifier.train(X, cv2.ml.ROW_SAMPLE, y)

        samples = np.array(X)
        labels = np.array(y)
        return samples, labels

    def predict(self, img):
        """.
        Given a single image of a character already cropped classify it.

        :img Image to classify
        
        """
        
        y = ... # Obtain the estimated label by the LDA + Bayes classifier

        return int(y)



