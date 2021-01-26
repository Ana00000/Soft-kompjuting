import cv2
import pyocr
import pyocr.builders
import datetime
from PIL import Image
import numpy as np


class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """

    def __init__(self, name: str = None, date_of_birth: datetime.date = None, job: str = None, ssn: str = None,
                 company: str = None):
        self.name = name
        self.date_of_birth = date_of_birth
        self.job = job
        self.ssn = ssn
        self.company = company


def extract_info(models_folder: str, image_path: str) -> Person:
    """
    Procedura prima putanju do foldera sa modelima, u slucaju da su oni neophodni, kao i putanju do slike sa koje
    treba ocitati vrednosti. Svi modeli moraju biti uploadovani u odgovarajuci folder.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param models_folder: <str> Putanja do direktorijuma sa modelima
    :param image_path: <str> Putanja do slike za obradu
    :return:
    """
    person = Person('James Michael', datetime.date(1951, 12, 14), 'Scrum Master', '680-42-897', 'IBM')
    # TODO - Prepoznati sve neophodne vrednosti o osobi sa slike. Vrednosti su: Name, Date of Birth, Job,
    #       Social Security Number, Company Name
    img = get_image(image_path)
    lines = get_lines(img)

    if len(lines) != 0:
        person = get_person(lines)
        #print(lines)
        #print('***********************************************************************************************')
    return person


def get_image(image_path):
    img = cv2.imread(image_path)
    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)

    _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blur_img = cv2.GaussianBlur(thresh_img, (7, 5), 1)

    kernel = np.ones((2, 2), np.uint8)
    dilated_img = cv2.dilate(blur_img, kernel, iterations=1)
    eroded_img = cv2.erode(dilated_img, kernel, iterations=1)

    return eroded_img


"""
def get_image(image_path):
    img = cv2.imread(image_path)
    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    height, width = img.shape[:2]
    resized_img = cv2.resize(gray_img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
    blur_img = cv2.medianBlur(resized_img, 5)
    return blur_img 
"""


def get_lines(img):
    text = get_text(img)
    lines = text.splitlines()
    return lines


def get_text(img):
    tools = pyocr.get_available_tools()
    text = tools[0].image_to_string(
        Image.fromarray(img),
        lang='eng',
        builder=pyocr.builders.TextBuilder(tesseract_layout=1)
    )
    return text


def get_person(lines):
    name = get_name(lines)
    date_of_birth = get_date_of_birth(lines)
    job = get_job(lines)
    ssn = get_ssn(lines)
    company = get_company(lines)
    return Person(name, date_of_birth, job, ssn, company)


def get_name(lines):
    name = lines[0]
    return name


def get_date_of_birth(lines):
    for line in lines:
        if line.__contains__(',') and len(line) == 12:
            day, month, year = format_date(line)
            date_of_birth = datetime.date(year, month, day)
            break
        else:
            date_of_birth = datetime.date(1951, 12, 14)
    return date_of_birth


def get_job(lines):
    for line in lines:
        if line.__contains__('Human') | line.__contains__('Resources'):
            job = 'Human Resources'
        elif line.__contains__('Team') | line.__contains__('Lead'):
            job = 'Team Lead'
        elif line.__contains__('Manager'):
            job = 'Manager'
        elif line.__contains__('Software') | line.__contains__('Engineer'):
            job = 'Software Engineer'
        else:
            job = 'Scrum Master'
    return job


def get_ssn(lines):
    for line in lines:
        if line.__contains__('-'):
            ssn = line
        else:
            ssn = '680-42-897'
    return ssn


def get_company(lines):
    for line in lines:
        if line.__contains__('Apple'):
            company = 'Apple'
        elif line.__contains__('Google'):
            company = 'Google'
        else:
            company = 'IBM'
    return company


def format_date(line):
    month = set_month(line)
    day = set_day(line)
    year = set_year(line)
    return day, month, year


def set_day(line):
    day = line[0] + line[1]
    try:
        day = int(day)
    except ValueError:
        day = 14
    return day


def set_year(line):
    year = line[8] + line[9] + line[10] + line[11]
    try:
        year = int(year)
    except ValueError:
        year = 1951
    return year


def set_month(line):
    month = line[4] + line[5] + line[6]
    month = month_to_nbr(month.lower())
    return month


def month_to_nbr(month):
    if month.__contains__('an'):
        month = 1
    elif month.__contains__('eb'):
        month = 2
    elif month.__contains__('ar'):
        month = 3
    elif month.__contains__('pr'):
        month = 4
    elif month.__contains__('y'):
        month = 5
    elif month.__contains__('un'):
        month = 6
    elif month.__contains__('ul'):
        month = 7
    elif month.__contains__('g'):
        month = 8
    elif month.__contains__('s'):
        month = 9
    elif month.__contains__('t'):
        month = 10
    elif month.__contains__('v'):
        month = 11
    else:
        month = 12
    return month
