import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1. change the directory in the main function starting from line 75
# 2. get the result in the result folder you have created named from '1.jpg',
# '2.jpg' to 'n.jpg'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# img is the original gray img
# mode 1 is for vertical histogram
# 	 2 is for horizontal histogram
def histogram(img, mode=2):
	ret = []
	[im_h, im_w] = img.shape
	if (mode == 1):
		ret = [0] * im_h
		for i in range(im_h):
			for j in range(im_w):
				if (img[i][j] == 0):
					ret[i] += 1
	elif (mode == 2):
		ret = [0] * im_w
		for j in range(im_w):
			for i in range(im_h):
				if (img[i][j] == 0):
					ret[j] += 1
	return ret

def divide( path ):
	# create result folder
	if (os.path.exists('result') == False):
		os.mkdir('result')

	for file in os.listdir('result'):
		os.remove(os.path.join('result', file))
	# load the example image
	image = cv2.imread(path)
	 
	# pre-process the image by resizing it, converting it to
	# graycale, blurring it, and computing an edge map
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	[ im_h, im_w ] = gray.shape
	# gray = gray[(int)(0.05 * im_h) : (int)(0.95 * im_h), (int)(0.05 * im_w) : (int)(0.95 * im_w)]
	# print(im_h, im_w)
	cv2.imwrite('gray.jpg', gray)
	ret, binary = cv2.threshold(gray ,98 , 255, cv2.THRESH_BINARY_INV)
	if (np.mean(gray) > 100):
		binary = ~binary
	cv2.imwrite('binary.jpg', binary)
	# get the historgram
	his = histogram(binary, mode = 2)
	# plot the histogram
	# plt.plot(his)
	# plt.show()

	# get the division lines
	lst = []
	# print(his, len(his))
	for i in range(len(his)):
		if (i == 0):
			continue
		if (his[i-1] >= 5 and his[i] < 5):
			lst.append(i)
		if (his[i-1] < 5 and his[i] >= 5):
			lst.append(i)
	# print(lst)

	# cut the images 
	base = (int)(0.05 * im_w)
	for i in range((int)(len(lst) / 2)):
		# 二值化图片输出
		x1 = lst[2*i] - 2
		x2 = lst[2*i+1] + 2
		cv2.imwrite('result/' + str(i) + '.jpg', image[:, x1: x2])
	# if (len(lst) % 2 == 1):
	# 	x1 = lst.pop() - 2
	# 	x2 = (int)(0.95 * im_w)
	# 	cv2.imwrite('result/' + str((int)(len(lst) /2 + 1)) + '.jpg', gray[:, x1: x2])

if __name__ == '__main__':
	divide('silk1.jpg')