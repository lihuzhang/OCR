from imutils.perspective import four_point_transform
from imutils import contours
from numpy import median
import imutils
import cv2
import numpy as np
import math
path = "1.jpg"
image = cv2.imread(path)
image2 = cv2.resize(image, (800, 800))
cv2.imwrite('right.jpg',image2)