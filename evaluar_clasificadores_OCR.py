# Asignatura de Visión Artificial (URJC). Script de evaluación.
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2023


import argparse
import os

#import panel_det
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
from sklearn import metrics

import lda_normal_bayes_classifier
from ocr_classifier import OCRClassifier


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y,x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument(
        '--classifier', type=str, default="", help='Classifier string name')
    parser.add_argument(
        '--train_path', default="./train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="./validation_ocr", help='Select the validation data dir')

    args = parser.parse_args()


    # 1) Cargar las imágenes de entrenamiento y sus etiquetas. 
    # También habrá que extraer los vectores de características asociados (en la parte básica 
    # umbralizar imágenes, pasar findContours y luego redimensionar)
    yt= []
    Xt= []
    ocr = OCRClassifier()
    ruta= args.train_path
    for dir in os.listdir(ruta):
        
        for img in os.listdir(ruta + "\\" + dir):
                img2 = cv2.imread(ruta + "\\" + dir + "\\" + img)
                if img2 is not None:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    contornos, _  = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contorno in contornos:
                        x, y, w, h = cv2.boundingRect(contorno)
                        # Recortar la región de interés (ROI) de la imagen
                        roi = img2[y:y + h, x:x + w]
                        yt.append(ocr.char2label(dir))
                        Xt.append(roi)
                
    # 2) Cargar datos de validación y sus etiquetas También habrá que extraer los vectores de características asociados (en la parte básica
    # umbralizar imágenes, pasar findContours y luego redimensionar)
    gt_labels = ...
    yv=[]
    Xv=[]
    ruta=args.validation_path
    for dir in os.listdir(ruta):

        for img in os.listdir(ruta + "\\" + dir):
            img2 = cv2.imread(ruta + "\\" + dir + "\\" + img)
            if img2 is not None:
                if len(img2.shape) > 2 and img2.shape[2] > 1:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                contornos, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contorno in contornos:
                    x, y, w, h = cv2.boundingRect(contorno)
                    # Recortar la región de interés (ROI) de la imagen
                    roi = img2[y:y + h, x:x + w]
                    yv.append(ocr.char2label(dir))
                    Xv.append(roi)

    gt_labels = np.array(yv)
    # 3) Entrenar clasificador
    lda = lda_normal_bayes_classifier.LdaNormalBayesClassifier((25, 25))
    lda.train(args.train_path)
    # 4) Ejecutar el clasificador sobre los datos de validación
    predicted_labels = []
    for img in Xv:
        img = cv2.resize(img, ocr.ocr_char_size)
        predicted_labels.append(lda.predict(img))


    # 5) Evaluar los resultados
    accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    sklearn.metrics
    print("Accuracy = ", accuracy)

