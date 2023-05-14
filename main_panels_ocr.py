import argparse
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
    lda = LdaNormalBayesClassifier()
    lda.train(args.train_path)
    # Create the OCR classifier

    # ocr = OCRClassifier()
    # Load testing data

    # Evaluate OCR over road panels





