import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
from imutils import face_utils
import argparse
import imutils
import dlib
import dlib_train

def load_image(img_path):
    return cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)


def extract_faces(img): #NOT WORKING
    # inicijalizaclija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # ucitavanje i transformacija slike
    image = img

    # NE TREBA MI GREYSCALE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detekcija svih lica na grayscale slici
    rects = detector(gray, 1)

    # iteriramo kroz sve detekcije korak 1.
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # odredjivanje kljucnih tacaka - korak 2
        shape = predictor(gray, rect)
        # shape predstavlja 68 koordinata
        shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
        print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
        print("Prva 3 elementa matrice")
        print(shape[:3])

        # konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ispis rednog broja detektovanog lica
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # crtanje kljucnih tacaka
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        plt.imshow(image)
        plt.show()



def main():

    # prikaz vecih slika
    matplotlib.rcParams['figure.figsize'] = 16, 12

    train_human = 'data/train/human/'
    train_anime = 'data/train/anime/'
    train_cartoon = 'data/train/cartoon/'

    human_imgs = []
    anime_imgs = []
    cartoon_imgs = []

    for img_name in os.listdir(train_human):
        img_path = os.path.join(train_human, img_name)
        img = load_image(img_path)
        print('human')
        human_imgs.append(img)
        extract_faces(img)
        break

    for img_name in os.listdir(train_anime):
        img_path = os.path.join(train_anime, img_name)
        img = load_image(img_path)
        print('anime')
        anime_imgs.append(img)
        extract_faces(img)


    for img_name in os.listdir(train_cartoon):
        img_path = os.path.join(train_cartoon, img_name)
        img = load_image(img_path)
        print('cartoon')
        cartoon_imgs.append(img)
        extract_faces(img)


if __name__ == "__main__":
    main()

