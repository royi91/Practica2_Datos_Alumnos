import argparse
import os
import cv2

import lda_normal_bayes_classifier

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    # Load training data
    lda = lda_normal_bayes_classifier.LdaNormalBayesClassifier((25, 25))
    lda.train(args.train_path)
    # Create the OCR classifier

    # Load testing data
    for img in os.listdir(args.test_path):
        img2 = cv2.imread(args.test_path + "\\" + img)
        if img2 is not None: lda.predict(img2)

    # Evaluate OCR over road panels
