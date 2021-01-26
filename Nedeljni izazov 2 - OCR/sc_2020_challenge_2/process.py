from __future__ import print_function

# numpy
import keras
import numpy as np

# cv
import cv2 as cv

# sklearn
from sklearn import datasets, metrics
from sklearn.cluster import KMeans

# fuzzywuzzy
import fuzzywuzzy

# theano keras
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

import os

"""
Code taken from practise 2 - OCR text
"""


def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
    putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija alfabeta
    :param serialization_folder: folder u koji treba sacuvati serijalizovani model
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju
    # Big letters
    colored_image_1 = load_image(train_image_paths[0])
    binary_image_1 = image_bin(image_gray(colored_image_1))
    kernel_11 = np.ones((2, 2))  # bigger
    dilated_image_11 = cv.dilate(invert(binary_image_1), kernel_11, iterations=2)
    kernel_2 = np.ones((3, 3))  # smaller
    eroded_image_1 = cv.erode(dilated_image_11, kernel_2, iterations=4)
    selected_regions_1, letters_1, _ = select_roi(colored_image_1.copy(), eroded_image_1)
    inputs_1 = prepare_for_ann(letters_1)
    # print('Number of detected regions:', len(letters_1))
    # cv.imwrite('big.png', selected_regions_1)

    # Small letters
    colored_image_2 = load_image(train_image_paths[1])
    binary_image_2 = image_bin(image_gray(colored_image_2))
    kernel_21 = np.ones((2, 2))  # bigger
    dilated_image_21 = cv.dilate(invert(binary_image_2), kernel_21, iterations=1)
    kernel_22 = np.ones((3, 3))  # smaller
    eroded_image_2 = cv.erode(dilated_image_21, kernel_22, iterations=5)
    selected_regions_2, letters_2, _ = select_roi(colored_image_2.copy(), eroded_image_2)
    inputs_2 = prepare_for_ann(letters_2)
    # print('Number of detected regions:', len(letters_2))
    # cv.imwrite('lil.png', selected_regions_2)

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h',
                'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    outputs = convert_output(alphabet)

    # Trained model - ann
    ann = load_trained_ann(serialization_folder)
    if ann is None:
        print("Training of the model started.")
        ann = create_ann()
        ann = train_ann(ann, inputs_1 + inputs_2, outputs)
        print("Training of the model finished.")
        serialize_ann(ann, serialization_folder)

    return ann


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string
    # Whole line of text
    """
    colored_image = load_image('.' + os.path.sep + 'dataset' + os.path.sep + 'validation' + os.path.sep + 'train0.png')
    gray_image = image_gray(colored_image)
    blur_image = cv.GaussianBlur(gray_image, (7, 5), 0)
    kernel = np.ones((7, 7))
    morph_img = cv.morphologyEx(blur_image, cv.MORPH_OPEN, kernel)
    threshold_img = cv.adaptiveThreshold(blur_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2)
    invert_img = invert(threshold_img)
    kernel = np.ones((2, 2))
    dilate_img = cv.dilate(invert_img, kernel, iterations=1)
    eroded_img = cv.erode(dilate_img, kernel, iterations=5)

    selected_regions, letters, distances = select_roi_text_white(invert_img.copy(), invert_img)
    """
    colored_image = load_image('.' + os.path.sep + 'dataset' + os.path.sep + 'validation' + os.path.sep + 'train2.png')
    gray_image = image_gray(colored_image)
    blur_image = cv.medianBlur(gray_image, 3)
    # threshold_img = cv.adaptiveThreshold(blur_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2)
    # invert_img = invert(threshold_img)
    ret, mask = cv.threshold(blur_image, 180, 255, cv.THRESH_BINARY)
    image_final = cv.bitwise_and(blur_image, blur_image, mask=mask)

    # kernel_1 = np.ones((4, 4))
    # kernel_2 = np.ones((3, 3))
    # dilate_img = cv.dilate(image_final, kernel_2, iterations=1)
    # eroded_img = cv.erode(dilate_img, kernel_1, iterations=7)

    selected_regions, letters, distances = select_roi_text_white(colored_image.copy(), image_final)

    print('Number of detected regions:', len(letters))
    cv.imwrite('regions.png', selected_regions)

    distances = np.array(distances).reshape(len(distances), 1)
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    try:
        k_means.fit(distances)
    except Exception as e:
        return ""

    inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))
    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h',
                'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    extracted_text = display_result(results, alphabet, k_means)
    print(extracted_text)
    return extracted_text


def select_roi_text_white(image_orig, invert_img):
    img, contours, hierarchy = cv.findContours(invert_img.copy(), cv.THRESH_TOZERO, cv.CHAIN_APPROX_NONE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if h > 80:
            region = invert_img[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 255), 5)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


def select_roi_text(image_orig, eroded_image):
    img, contours, hierarchy = cv.findContours(eroded_image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if 45 < h and 20 < w:
            region = eroded_image[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 255), 5)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


def load_image(path):
    return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)


def image_gray(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def image_bin(image_gs):
    ret, image_binary = cv.threshold(image_gs, 120, 255, cv.THRESH_TOZERO)
    return image_binary


def invert(image):
    return 255 - image


def dilate(image):
    kernel = np.ones((3, 3))
    return cv.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))
    return cv.erode(image, kernel, iterations=3)


def resize_region(region):
    resized = cv.resize(region, (28, 28), interpolation=cv.INTER_NEAREST)
    return resized


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann


def convert_output(outputs):
    return np.eye(len(outputs))


def select_roi(image_orig, eroded_image):
    img, contours, hierarchy = cv.findContours(eroded_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if 40 < h:
            region = eroded_image[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 255), 5)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


def create_ann():
    ann = Sequential()
    ann.add(Dense(128 * 2, input_dim=28 * 28, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train_ann(ann, x_train, y_train):
    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.float32)

    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(x_train, y_train, epochs=750, batch_size=1, verbose=0, shuffle=False)

    return ann


def serialize_ann(ann, serialization_folder):
    model_json = ann.to_json()
    with open(serialization_folder + "trainedModel.json", "w") as json_file:
        json_file.write(model_json)
    ann.save_weights(serialization_folder + "trainedModel.h5")


def load_trained_ann(serialization_folder):
    try:
        json_file = open(serialization_folder + 'trainedModel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        ann.load_weights(serialization_folder + "trainedModel.h5")
        print("Trained model successfully loaded.")
        return ann
    except Exception as e:
        return None


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet, k_means):
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result
