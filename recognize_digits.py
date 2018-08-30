# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
from numpy import median
import imutils
import cv2
import numpy as np

def filter_rois( rois ):
	rets_raw = []
	for i in range(len(rois)):
		add = True
		for j in range(len(rois)):
			if( i == j ):
				continue
			if( rois[i][0] > rois[j][0] and rois[i][1] > rois[j][1] 
				and ( rois[i][0] + rois[i][2] ) < ( rois[j][0]+ rois[j][2] ) 
				and ( rois[i][1] + rois[i][3] ) < ( rois[j][1]+ rois[j][3] )):
				add = False
				print("i", i, "j", j)
				break
				#this contour is contained in another one
		if(add):
			rets_raw.append(rois[i])
	def get_width(x): return x[2]
	def get_height(x): return x[3]
	widths = list(map(get_width, rets_raw))
	heights = list(map(get_height, rets_raw))
	median_width = median( widths )
	median_height = median( heights )
	rets = list(filter(lambda x: ( (x[3] < 2 * median_height) and (x[3] > median_height/2) ), rets_raw))
	return rets

def main( path ):
	# load the example image
	image = cv2.imread(path)
	 
	# pre-process the image by resizing it, converting it to
	# graycale, blurring it, and computing an edge map
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	[ im_h, im_w ] = gray.shape
	blurred = cv2.GaussianBlur(gray, (5, 5), 1)

	v = np.median(blurred)
	sigma=0.33
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(blurred, lower, upper)
	cv2.imshow('edged',edged)
	cv2.imshow('blurred',blurred)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# find contours in the edge map, then sort them by their
	# size in descending order
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts = sorted(cnts, 
			key=lambda c: cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3], 
			reverse = True)
	# area_threshold = edged.size / 200
	# cnts = [ c for c in cnts if cv2.contourArea(c) > area_threshold ]
	#cv2.drawContours(image, cnts, -1, (0,255,0), 3 )
	screens = []
	orig_screens = []
	screens_bound = []
	# loop over the contours
	image_h = gray.shape[0]
	image_w = gray.shape[1]
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		(x, y, w, h) = cv2.boundingRect(c)
		close_to_edge = True
		for pnt in c:
			if( (pnt[0][0] - x) < w/10 or (pnt[0][0] - x) > w*9/10
				or (pnt[0][1] - y) < h/5 or (pnt[0][1] - y) > h*4/5 ):
				continue
			else:
				close_to_edge = False
				break
		if not close_to_edge:
			continue
		img = image.copy()
		cv2.drawContours(img, [approx], -1, (0,255,0), 3 )
		# if( w > 500 and h > 70 ):
		# 	cv2.imshow('cnt_%d' %i,roi)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		# if the contour has four vertices, then we have found
		# the thermostat display
		
		if (w + h) > (image_w + image_h)/30:
			print("bound", int(x+w/20), int(y+h/20), int(w*9/10), int(h*9/10))
			screens_bound.append( [int(x+w/20), int(y+h/20), int(w*9/10), int(h*9/10)] )
			
	# extract the thermostat display, apply a perspective transform to it
	# print(displayCnt)
	# cv2.imshow('screens',screens[0])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	screens_bound = filter_rois(screens_bound)
	print("num screens", len(screens_bound))

	for [x, y, w, h] in screens_bound:
		print("bound", x, y, x+w, y+h)
		roi = gray[y:y+h, x:x+w]
		orig_roi = image[y:y+h, x:x+w]
		roi = imutils.resize(roi, height=5*h)
		orig_roi = imutils.resize(orig_roi, height=5*h)
		screens.append(roi)
		orig_screens.append(orig_roi)
	result = ""
	for i in range( len(screens) ):
		warped = screens[i]
		output = orig_screens[i]
		half_param = warped.shape[0] + warped.shape[1]
		# thresh = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		#             cv2.THRESH_BINARY,21,5)
		thresh_nongaus = cv2.threshold(warped, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		thresh_nongaus = cv2.morphologyEx(thresh_nongaus, cv2.MORPH_OPEN, kernel)

		#dilation_kernel = np.ones((half_param/130,half_param/130),np.uint8) 
		#thresh_nongaus2 = cv2.erode(thresh_nongaus, dilation_kernel)
		# find contours in the thresholded image, then initialize the
		# digit contours lists
		cnts = cv2.findContours(thresh_nongaus.copy(), cv2.RETR_TREE,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		digitCnts = []
		cv2.imshow('thresh_nongaus',thresh_nongaus)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		# loop over the digit area candidates
		for c in cnts:
			# compute the bounding box of the contour
			(x, y, w, h) = cv2.boundingRect(c)
			# if the contour is sufficiently large, it must be a digit
			if (h >= 0.05 * image_h or w >= 0.05 * image_w) and ( h <= 0.5 * image_h or w <= 0.5 * image_w):
				digitCnts.append(c)
		print("len digitcnts", len(digitCnts))
		
		# sort the contours from left-to-right, then initialize the
		# actual digits themselves
		# cv2.drawContours(output, digitCnts, -1, (0,255,0), 3)
		# cv2.imshow('thresh_cnt',output)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
		digits = []
		rois = []
		# loop over each of the digits
		for c in digitCnts:
			# extract the digit ROI
			(x, y, w, h) = cv2.boundingRect(c)
			rois.append( [x, y, w, h] )

		rois = filter_rois(rois)
		for [x, y, w, h] in rois:
			roi = thresh_nongaus[y:y + h, x:x + w]
			cv2.imwrite('./curr_pic/roi_%d.png'%i,roi)
			############call your function here############
			############use roi as input############
			############return a digit as string named single_digit############
			#result += single_digit

	result = result[:3]+"\n" + result[3:] #for this machine result should have exactly 6 digits
	return(result)

if __name__ == '__main__':
	main("2.jpg")