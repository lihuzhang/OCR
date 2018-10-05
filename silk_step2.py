# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
from numpy import median
import imutils
import cv2
import numpy as np
import math

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

def get_consecutive_trues( arr ):
	result = []
	tmp_start = 0
	tmp_end = 0
	for i in range(len(arr)):
		if (i == 0 and arr[i]) or (arr[i] > arr[i-1]):
			tmp_start = i
		elif (i == len(arr)-1 and arr[i]) or (arr[i] < arr[i-1]):
			tmp_end = i
			result.append([tmp_start, tmp_end])
	return result

def overlay( arr1, arr2 ):
	left = arr2
	right = arr1
	if( arr1[0] < arr2[0] ):
		left = arr1
		right = arr2
	if( left[2] < right[0] ):
		return False
	if( left[3] < right[1] ):
		return False
	if( left[1] > right[3] ):
		return False
	return True

def calc_dist( mainarr, newarr ):
	new_center = [ (newarr[0] + newarr[2])/2, (newarr[1] + newarr[3])/2]
	candidate = []
	if( ( (new_center[0] > mainarr[0]) and (new_center[0] < mainarr[2]) )
		or ( (new_center[1] > mainarr[1]) and (new_center[1] < mainarr[3]) ) ):
		candidate = [ abs( new_center[0] - mainarr[0] ), abs( new_center[0] - mainarr[2] ),
					  abs( new_center[1] - mainarr[1] ), abs( new_center[1] - mainarr[3] )]
	else:
		candidate = [ math.sqrt( (new_center[0] - mainarr[0])**2 + (new_center[1] - mainarr[1])**2 ),
					  math.sqrt( (new_center[0] - mainarr[0])**2 + (new_center[1] - mainarr[3])**2 ),
					  math.sqrt( (new_center[0] - mainarr[2])**2 + (new_center[1] - mainarr[1])**2 ),
					  math.sqrt( (new_center[0] - mainarr[2])**2 + (new_center[1] - mainarr[3])**2 )]
	return min(candidate)

def combine_cnts( cnts ):
	result = []
	for new_c in cnts:
		if( not result ):
			result.append(new_c)
		overlay_list = []
		for i in range(len(result)):
			if( overlay( new_c, result[i] ) ):
				overlay_list.append(i)
		if not overlay_list:
			result.append(new_c)
		else:
			min_ind = 0
			dist = calc_dist( result[overlay_list[0]], new_c )
			for j in range(1, len(overlay_list)):
				new_dist = calc_dist( result[overlay_list[j]], new_c )
				if( new_dist < dist ):
					dist = new_dist
					min_ind = j
			result[min_ind] = [min(result[min_ind][0], new_c[0]),min(result[min_ind][1], new_c[1]),
				 max(result[min_ind][2], new_c[2]),max(result[min_ind][2], new_c[2])]
	return result


def main( path ):
	# load the example image
	image = cv2.imread(path)
	#gray = image[:,:,1]+image[:,:,2]
	# pre-process the image by resizing it, converting it to
	# graycale, blurring it, and computing an edge map
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	[ im_h, im_w ] = gray.shape
	gray = gray[int(im_h/40):int(im_h*39/40),int(im_w/40):int(im_w*39/40)]
	image = image[int(im_h/40):int(im_h*39/40),int(im_w/40):int(im_w*39/40)]
	[ im_h, im_w ] = gray.shape

	blurred = cv2.GaussianBlur(gray, (3, 3), 1)
	v = np.median(blurred)
	sigma=0.5
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(blurred, lower, upper)
	# cv2.imshow('gray',gray)
	# cv2.imshow('edged',edged)
	# cv2.imshow('blurred',blurred)
	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# cv2.drawContours(image, cnts, -1, (0,255,0), 3)
	# cv2.imshow('image',image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	thresh_nongaus = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	thresh_nongaus = cv2.morphologyEx(thresh_nongaus, cv2.MORPH_OPEN, kernel)
	cv2.imshow('thresh_nongaus',thresh_nongaus)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

		# if (w + h) > (image_w + image_h)/30:
		# 	print("bound", int(x+w/20), int(y+h/20), int(w*9/10), int(h*9/10))
		# 	screens_bound.append( [int(x+w/20), int(y+h/20), int(w*9/10), int(h*9/10)] )
			
	# extract the thermostat display, apply a perspective transform to it
	# print(displayCnt)
	# cv2.imshow('screens',screens[0])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#screens_bound = filter_rois(screens_bound)
	horizontal_num_black_pts = [ np.sum(255-x) for x in thresh_nongaus ]
	hist = np.histogram(horizontal_num_black_pts, len(horizontal_num_black_pts))
	#print(hist)
	avg_black_points = sum(horizontal_num_black_pts)/len(horizontal_num_black_pts)
	avg_black_points = avg_black_points/2
	horizontal_black_lines = [ True if x > avg_black_points else False for x in horizontal_num_black_pts ]
	consec_trues = get_consecutive_trues(horizontal_black_lines)
	consec_trues = [x for x in consec_trues if (x[1]-x[0]) > 1 ]
	print(consec_trues)

	screens_bound = [ [0, x[0], im_w, (x[1]-x[0]+1)] for x in consec_trues ]
	screens = []
	orig_screens = []
	print("num lines", len(consec_trues))
	print("lines", consec_trues)

	for [x, y, w, h] in screens_bound:
		print("bound", x, y, x+w, y+h)
		roi = gray[y:y+h, x:x+w]
		orig_roi = image[y:y+h, x:x+w]
		# roi = imutils.resize(roi, height=5*h)
		# orig_roi = imutils.resize(orig_roi, height=5*h)
		screens.append(roi)
		orig_screens.append(orig_roi)

	result = ""
	for i in range( len(screens) ):
		dilation_kernel1 = np.array([[0,1,0],[0,0,0],[0,1,0]],np.uint8)
		erode_kernel1 = np.array([[0,0,0,0,0],[1,1,0,1,1],[0,0,0,0,0]],np.uint8)
		dilation_kernel2 = np.array([[0,1,0],[1,0,1],[0,1,0]],np.uint8)
		thresh_nongaus2 = cv2.dilate(screens[i], dilation_kernel1,iterations = int(len(screens[i])/30))
		thresh_nongaus2 = cv2.dilate(thresh_nongaus2, dilation_kernel2,iterations = int(len(screens[i])/30))
		cv2.imshow('screen',screens[i])
		cv2.imshow('thresh_nongaus2',thresh_nongaus2)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		# find contours in the thresholded image, then initialize the
		# digit contours lists
		# blurred = cv2.GaussianBlur(thresh_nongaus2, (5, 5), 1)
		# v = np.median(blurred)
		# sigma=0.33
		# lower = int(max(0, (1.0 - sigma) * v))
		# upper = int(min(255, (1.0 + sigma) * v))
		# edged = cv2.Canny(blurred, lower, upper)

		# cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
		# 	cv2.CHAIN_APPROX_SIMPLE)
		# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		# cv2.drawContours(orig_screens[i], cnts, -1, (0,255,0), 3)
		# cv2.imshow('orig_screens',orig_screens[i])
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		allcnts = []
		# loop over the digit area candidates
		for c in cnts:
			# compute the bounding box of the contour
			(x, y, w, h) = cv2.boundingRect(c)
			if( w*h > im_h*im_w/100 ):
				allcnts.append( [x, y, x+w, y+h] )
			# if the contour is sufficiently large, it must be a digit
		allcnts = combine_cnts( allcnts )
		print("len digitcnts", len(allcnts))
		
		# sort the contours from left-to-right, then initialize the
		# actual digits themselves
		# cv2.drawContours(output, digitCnts, -1, (0,255,0), 3)
		# cv2.imshow('thresh_cnt',output)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		digits = []
		rois = []
		# loop over each of the digits
		for bound in allcnts:
			# extract the digit ROI
			rois.append( bound )

		#rois = filter_rois(rois)
		trash = 0
		for [xmin, ymin, xmax, ymax] in rois:
			roi = image[ymin:ymax, xmin:xmax]
			cv2.imshow('roi%d'%trash,roi)
			trash +=1
		cv2.waitKey(0)
		cv2.destroyAllWindows()

			#cv2.imwrite('./curr_pic/roi_%d.png'%i,roi)
			############call your function here############
			############use roi as input############
			############return a digit as string named single_digit############
			#result += single_digit

	result = result[:3]+"\n" + result[3:] #for this machine result should have exactly 6 digits
	return(result)

if __name__ == '__main__':
	main("silk3.jpg")