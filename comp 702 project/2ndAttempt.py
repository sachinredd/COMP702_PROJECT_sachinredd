import glob

import cv2
import mahotas as mt
import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


def preprocessing(image):
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = scale_image(processed_image)
    processed_image = to_grayscale(processed_image)
    processed_image = histogram_equalize(processed_image)
    processed_image = binary_thresholding(processed_image)
    processed_image = median_filter(processed_image,3)
    #processed_image = average_smoothing(processed_image)
    #processed_image = canny_edge_detection(processed_image)
    processed_image = gaussian_blur(processed_image)
    return processed_image
def scale_image(image):
    return cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def histogram_equalize(image):
    return cv2.equalizeHist(image)
def median_filter(image,num):
    return cv2.medianBlur(image, num)
def average_smoothing(image):
    return cv2.blur(src=image, ksize=(3, 3))
def gaussian_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)
def binary_thresholding(image):
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
def haralick(image):
    textures = mt.features.haralick(image)
    return textures.mean(axis=0)
def canny_edge_detection(image):
    edges = cv2.Canny(image, threshold1=100, threshold2=200, L2gradient=True)
    image = cv2.bitwise_and(image, image, edges)
    return image
def region_segmentation(image):
    x = len(image)
    y = len(image)
    item = {
        'top left': [0,round(y/2),0,round(x/2)],
        'top rigth':[0,round(y/2),round(x/2),x],
        'low left' :[round(y/2),0,round(x/2),0],
        'low right':[round(y/2),y,round(y/2),x]
    }
    image_regions= []
    discriminant_subets = []
    for elements in item.items():
        region = elements[1]
        image_regions.append(image[region[0]:region[1], region[2]:region[3]])
        discriminant_subets.append(image[region[0]:region[1], region[2]:region[3]])

    return image_regions, discriminant_subets
def getTrain():
    x_train = []
    y_train = []
    info = ''
    info1 = "train/1"
    info2 = "train/2"
    info5 = "train/5"
    info10 = "train/10"
    info20 = "train/20"
    info50 = "train/50"

    for i in range(6):
        match i:
            case 0: info = info1
            case 1: info = info2
            case 2: info = info5
            case 3: info = info10
            case 4: info = info20
            case 5: info = info50

        path = glob.glob((info + "/*.jpg"))+glob.glob((info + "/*.png"))+glob.glob((info + "/*.jpeg"))
        for img in path:
            processed_image = cv2.imread(img)
            processed_image = preprocessing(processed_image)
            canny_segmentation = canny_edge_detection(processed_image)
            discrim_regions = region_segmentation(canny_segmentation)[1]
            haralick = haralick(discrim_regions[0])
            x_train.append(haralick)
            y_train.append(info)

    return x_train, y_train
def getTest():
    x_test= []
    test_path = "test"

    yl = []
    path = glob.glob((test_path + "/*.jpg")) + glob.glob((test_path + "/*.png")) + glob.glob((test_path + "/*.jpeg"))
    for file1 in path:
        if file1.find('10c')>0:
            ylbl = "train/10"
        elif file1.find('R1')>0:
            ylbl = "train/1"
        elif file1.find('R2')>0:
            ylbl = "train/2"
        elif file1.find('R5')>0:
            ylbl = "train/5"
        elif file1.find('20c')>0:
            ylbl = "train/20"
        else:
            ylbl = "train/50"
        yl.append(ylbl)
        image = cv2.imread(file1)
        processed_image = preprocessing(image)
        canny_segmentation = canny_edge_detection(processed_image)
        discrim_regions = region_segmentation(canny_segmentation)[1]
        haralick = haralick(discrim_regions[0])

        x_test.append(haralick)  # Append which feature extraction method you one you want
    return x_test,yl







def svc(x_train, y_train, test_x):
    clf_svm = SVC(max_iter=50000, random_state=9)
    clf_svm.fit(x_train, y_train)
    predictions = [clf_svm.predict([testing_image])[0] for testing_image in test_x]
    return predictions

def svm(x_train, y_train, test_x):
    clf_svm = LinearSVC(max_iter=50000, random_state=9)
    clf_svm.fit(x_train, y_train)
    predictions = [clf_svm.predict([testing_image])[0] for testing_image in test_x]
    return predictions
def k_nearest_classifier(no_of_neighbours, train_x, train_y, test_x):
    knn_model = KNeighborsClassifier(no_of_neighbours)
    knn_model.fit(train_x, train_y)
    predictions = [knn_model.predict([testing_image])[0] for testing_image in test_x]
    return predictions
def kn(num, x_train, y_train, x_test,y_test):
    knn_classification_predictions = k_nearest_classifier(num, x_train, y_train, x_test)
    print(f"{knn_classification_predictions}")
    print(classification_report(y_test, knn_classification_predictions))
def mlp(x_train, y_train,x_test):
    clf = MLPClassifier(hidden_layer_sizes=(10, 40),random_state=9,learning_rate_init=0.1)
    clf.fit(x_train, y_train)
    predictions = [clf.predict([testing_image])[0] for testing_image in x_test]
    return predictions






x_train,y_train = getTrain()
x_test,y_test = getTest()


print("------------Linear SVM--------------")
svm_predictions = svm(x_train, y_train, x_test)
print(f"{svm_predictions}")
print(classification_report(y_test, svm_predictions))

print("------------SVC--------------")
svm_predictions = svc(x_train, y_train, x_test)
print(f"{svm_predictions}")
print(classification_report(y_test, svm_predictions))

print("----------KN--------------")
kn(1,x_train, y_train, x_test,y_test)

print("--------MLP--------------------")
mlp(x_train,y_train,x_test)
mlp_predictions = mlp(x_train, y_train, x_test)
print(f"{mlp_predictions}")
print(classification_report(y_test, mlp_predictions))

