import cv2
import sys
import os
import timeit
import face_recognition as fr
import pickle
import numpy as np
from skimage import io as io
from tqdm import tqdm
from sklearn import datasets, svm, metrics
from random import randint
import random

CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)


"""
    detect_faces
    description: extract faces from image capture
"""
def detect_faces(image_path):
    image = cv2.imread(image_path)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0)

    faces_path = []
    for x, y, w, h in faces:
        sub_img = image[y - 10:y + h + 10, x - 10:x + w + 10]
        os.chdir("extracted")
        path = str(randint(0, 10000)) + ".jpg"
        cv2.imwrite(path, sub_img)

        faces_path.append("./extracted/" + path)

        os.chdir("../")
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    return faces_path


"""
    initialize
    description: load model generated in separate script
"""
def initialize():
    with open('./gender_data.pkl', 'rb') as f:
        gender_dataset_raw = pickle.load(f)

    random.shuffle(gender_dataset_raw)

    embedding_list_train = list()
    gender_label_list_train = list()

    embedding_list_test = list()
    gender_label_list_test = list()

    for emb, label in gender_dataset_raw[:248]:
        embedding_list_train.append(emb)
        gender_label_list_train.append(label)

    for emb, label in gender_dataset_raw[:248]:
        embedding_list_test.append(emb)
        gender_label_list_test.append(label)

    classifier = svm.SVC(gamma='auto', kernel='rbf', C=20)
    classifier.fit(embedding_list_train, gender_label_list_train)

    return classifier


"""
    detect_gender
    description: detect sex by loading the image to the generated classifier at the function initialize
"""
def detect_gender(image_path, classifier):
    img = io.imread(image_path)
    img_encoded = fr.face_encodings(img)[0]

    return classifier.predict([img_encoded])


"""
    start
    description: take picture, send it to detect_faces to extract faces, send faces_path to detect_gender
                    to detect sex
"""
def start(classifier):
    temp_path = "./alon.jpg"

    faces_path = detect_faces(temp_path)

    print("log: ", faces_path)

    for face in faces_path:
        result = detect_gender(face, classifier)

        print(result)


if __name__ == '__main__':
    classifier = initialize()
    start(classifier)