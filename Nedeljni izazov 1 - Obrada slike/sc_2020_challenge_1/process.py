import cv2
import numpy as np


def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca, belih krvnih zrnaca i
    informaciju da li pacijent ima leukemiju ili ne, na osnovu odnosa broja krvnih zrnaca

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca,
             <int> broj prebrojanih belih krvnih zrnaca,
             <bool> da li pacijent ima leukemniju (True ili False)
    """

    # TODO - Prebrojati crvena i bela krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure
    """
        White cells
    """
    # Getting image
    white_cells_img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(white_cells_img, cv2.COLOR_BGR2GRAY)

    # Apply median filter for smoothing
    smooth_img_white = cv2.medianBlur(gray_img, 5)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    closing_img = cv2.morphologyEx(smooth_img_white, cv2.MORPH_CLOSE, kernel)

    # Adaptive threshold gaussian filter
    threshold_img = cv2.adaptiveThreshold(closing_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 9, 2)

    # Segmentation of white cells
    circles_a = cv2.HoughCircles(threshold_img, cv2.HOUGH_GRADIENT, 1.2, 105,
                                 param1=50, param2=28, minRadius=2, maxRadius=28)

    # Getting count of white cells
    cell_count_a = []
    if circles_a is not None:
        circles_a = np.round(circles_a[0, :]).astype("int")
        for (r) in circles_a:
            cell_count_a.append(r)
    # print(len(cell_count_a))
    white_blood_cell_count = len(cell_count_a)

    """
          Red cells
    """
    # Getting image
    red_cells_img = cv2.imread(image_path)

    # Getting red color
    red = [(150, 137, 168), (218, 209, 208)]  # (lower), (upper)
    colors = [red]

    # Apply median filter for smoothing
    smooth_img_red = cv2.medianBlur(red_cells_img, 3)

    cell_count_b = 0
    output = red_cells_img.copy()
    for lower, upper in colors:
        mask = cv2.inRange(smooth_img_red, lower, upper)

        # Segmentation of red cells
        circles_b = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=15, param2=17,
                                     minRadius=2, maxRadius=60)

        # Getting count of red cells
        if circles_b is not None:
            circles_b = np.round(circles_b[0, :]).astype("int")

            for (x, y, r) in circles_b:
                cv2.circle(output, (x, y), r, (255, 0, 255), 2)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 255), -1)
                cell_count_b += 1

    # cv2.imwrite('output.png', output)
    # print(cell_count_b)
    red_blood_cell_count = cell_count_b

    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu
    #  vrednost ove procedure

    if (white_blood_cell_count > 2
            or
            white_blood_cell_count >= (red_blood_cell_count / 3)):
        has_leukemia = True
    else:
        has_leukemia = False

    return red_blood_cell_count, white_blood_cell_count, has_leukemia