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


def image_to_feature_vector(image, size=(32, 32)):
    #output feature vector will be a list of 32 x 32 x 3 = 3,072 numbers.
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()


train_rawImages = []
train_features = []
train_labels = []
def attach_information_images(datapath,label):
    i = 0
    for img_name in os.listdir(datapath):
        i = i + 1
        img_path = os.path.join(datapath, img_name)
        img = load_image(img_path)
        train_labels.append(label)
        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        train_rawImages.append(image_to_feature_vector(img))
        train_features.append(extract_color_histogram(img))
        if i > 0 and i % 200 == 0:
            print("[INFO] processed " + str(i) + " humans")
        # plt.imshow(img)
        # plt.show()


def train_knn():

    train_human = 'data/train/human/'
    train_anime = 'data/train/anime/'
    train_cartoon = 'data/train/cartoon/'

    attach_information_images(train_human,'human')
    attach_information_images(train_anime,'anime')
    attach_information_images(train_cartoon,'cartoon')

    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        train_rawImages, train_labels, test_size=0.25, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        train_features, train_labels, test_size=0.25, random_state=42)

    # krece trening raw
    neighbors=1
    jobs=-1 #upotrebljavamo sva jezgra procesora computing power goes brrrr
    print("[INFO] evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=neighbors,
                                 n_jobs=jobs)
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

    # trening histogrami
    # train and evaluate a k-NN classifer on the histogram
    # representations
    print("[INFO] evaluating histogram accuracy...")
    model = KNeighborsClassifier(n_neighbors=neighbors,
                                 n_jobs=jobs)
    model.fit(trainFeat, trainLabels)
    acc = model.score(testFeat, testLabels)
    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


def main():

    # prikaz vecih slika
    matplotlib.rcParams['figure.figsize'] = 16, 12

    print('Enter option:\n')
    print('Train = 0; Test = 1')
    option=int(input())
    if(option!=0):
        #test
        print('nema josh')
    else:
        train_knn()





if __name__ == "__main__":
    main()

