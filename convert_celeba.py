import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CASCADE_PATH = os.path.join(os.getenv("HOME"),
                            ".pyenv/versions/anaconda-4.0.0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
DATA_PATH = os.path.join(os.getenv("HOME"), "share/data/")
ATTRIBUTES_PATH = os.path.join(DATA_PATH, 'list_attr_celeba.txt')
IMAGES_PATH = os.path.join(DATA_PATH, 'img_align_celeba')

if os.path.exists(CASCADE_PATH) is False:
    print "%s is not exist" % CASCADE_PATH
    sys.exit()
cascade = cv2.CascadeClassifier(CASCADE_PATH)
df = pd.read_csv(ATTRIBUTES_PATH, sep=' ', skiprows=1)
df = np.array(df)

color = (255, 255, 255)
expansion = 1.3
img_size = (64, 64)
images = []
attributes = []

for attribute in tqdm(df):
    try:
        img = cv2.imread(os.path.join(IMAGES_PATH, attribute[0]))
        attribute = attribute[1:]
    except:
        sys.exit()

    img_gray = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.2,
                                        minNeighbors=2, minSize=(10, 10))

    if len(facerect) > 0:
        for rect in facerect:
            length = (np.array(rect[0:2] + rect[2:4]) - np.array(rect[0:2]))[0]
            length = length * expansion - length
            leftup = np.array(rect[0:2]) - \
                np.array([length / 2, length / 2], dtype=int)
            rightdown = np.array(rect[0:2] + rect[2:4]) + \
                np.array([length / 2, length / 2], dtype=int)

            if not (leftup[0] < 0 or leftup[1] < 0 or rightdown[0] < 0 or rightdown[1] < 0):
                # cv2.rectangle(img, tuple(leftup),tuple(rightdown), color, thickness=2)
                try:
                    _img = img[leftup[1]:rightdown[1], leftup[0]:rightdown[0]]
                except:
                    print "too large range", np.array(img).shape, leftup, rightdown

                try:
                    # complete to create image
                    plt_img = Image.fromarray(_img[:, :, ::-1])
                    face_size = np.array(np.array(plt_img).shape[0],
                                         dtype=np.float32) / img.shape[0]
                    if face_size > 0.4:
                        resize_img = plt_img.resize(img_size)
                        images.append(np.array(resize_img))
                        attributes.append(attribute)
                    else:
                        print "too short face size", face_size

                except:
                    print "failed to resize image", img.shape
            else:
                print "invalid image range"

images = np.array(images)
attributes = np.array(attributes)
print images.shape
print attributes.shape

np.save('celeba_images', images)
np.save('celeba_attributes', attributes)
