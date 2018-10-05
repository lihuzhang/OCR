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
			continue
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
			result[overlay_list[min_ind]] = [min(result[overlay_list[min_ind]][0], new_c[0]),min(result[overlay_list[min_ind]][1], new_c[1]),
				 max(result[overlay_list[min_ind]][2], new_c[2]),max(result[overlay_list[min_ind]][3], new_c[3])]
	return result

def main( path, heat_pic_path, point_of_interest ):
	# load the example image
	image = cv2.imread(path)
	#image = cv2.resize(image,(500,500))
	[ im_h, im_w, trash ] = image.shape
	image_heat = cv2.imread(heat_pic_path)
	image_heat = cv2.resize(image_heat,(im_w,im_h))
	[ im_h, im_w, trash ] = image_heat.shape
	
	image_rb = cv2.cvtColor(image_heat, cv2.COLOR_BGR2GRAY);
	avg = np.average(image_rb)
	image_rb = cv2.inRange(image_rb, avg*2, 255)
	cv2.imshow('image_rb',image_rb)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(im_h/50), int(im_w/50)))
	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	thresh_nongaus = cv2.morphologyEx(image_rb, cv2.MORPH_OPEN, kernel)
	#thresh_nongaus = cv2.threshold(thresh_nongaus, 127, 255, cv2.THRESH_BINARY)[1]
	blurred_heat = cv2.GaussianBlur(thresh_nongaus, (5, 5), 1)
	v = np.median(blurred_heat)
	sigma=0.33
	lower_heat = int(max(0, (1.0 - sigma) * v))
	upper_heat = int(min(255, (1.0 + sigma) * v))
	edged_heat = cv2.Canny(blurred_heat, lower_heat, upper_heat)
	#ret, markers = cv2.connectedComponents(thresh_nongaus)
	#markers = cv2.watershed(image, markers)==-1
	#image_copy //= 2
	#image[markers] = (255,0,0)
	cv2.imshow('thresh_nongaus',thresh_nongaus)
	cv2.imshow('edged',edged_heat)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cnts_heat = cv2.findContours(edged_heat.copy(), cv2.RETR_LIST,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts_heat = cnts_heat[0] if imutils.is_cv2() else cnts_heat[1]
	cap = []
	all_cnt = []
	# loop over the contours
	for c_heat in cnts_heat:
		# approximate the contour
		peri = cv2.arcLength(c_heat, True)
		approx = cv2.approxPolyDP(c_heat, 0.02 * peri, True)
		(x, y, w, h) = cv2.boundingRect(c_heat)
		all_cnt.append([x,y,x+w,y+h])
	all_cnt = combine_cnts( all_cnt  )
	print("all_cnt",len(all_cnt))
	image_copy = image.copy()
	cv2.drawContours(image_copy, cnts_heat, -1, (0,255,0), 3)
	cv2.imshow('image_heat_cnt',image_copy)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	blurred = cv2.GaussianBlur(image, (5, 5), 1)
	v = np.median(blurred)
	sigma=0.33
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(blurred, lower, upper)

	num_write = 0
	for area in all_cnt:
		area_image = image[area[1]:area[3],area[0]:area[2]]
		cv2.imwrite('area_image_%d.png'%num_write,area_image)
		num_write += 1
	# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# selected = []
	# for c in cnts:
	# 	# approximate the contour
	# 	peri = cv2.arcLength(c, True)
	# 	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# 	(x, y, w, h) = cv2.boundingRect(c)
	# 	if( (w*h)>100 and x >= xlim and y >= ylim and (x+w) <= (xlim+wlim) and (y+h) <= (ylim + hlim) ):
	# 		selected.append(c)
	# image_copy = image.copy()
	# cv2.drawContours(image_copy, selected, -1, (0,255,0), 3)
	# cv2.imshow('image_selected_cnt',image_copy)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# cv2.imwrite('image_selected_cnt.png',image_copy)
	# extract the thermostat display, apply a perspective transform to it
	# print(displayCnt)
	# cv2.imshow('screens',screens[0])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#screens_bound = filter_rois(screens_bound)
	# print("num screens", len(screens_bound))

	# for [x, y, w, h] in screens_bound:
	# 	print("bound", x, y, x+w, y+h)
	# 	roi = gray[y:y+h, x:x+w]
	# 	orig_roi = image[y:y+h, x:x+w]
	# 	roi = imutils.resize(roi, height=5*h)
	# 	orig_roi = imutils.resize(orig_roi, height=5*h)
	# 	screens.append(roi)
	# 	orig_screens.append(orig_roi)
	# result = ""
	# for i in range( len(screens) ):
	# 	warped = screens[i]
	# 	output = orig_screens[i]
	# 	half_param = warped.shape[0] + warped.shape[1]
	# 	# thresh = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	# 	#             cv2.THRESH_BINARY,21,5)
	# 	thresh_nongaus = cv2.threshold(warped, 0, 255,
	# 		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	# 	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	# 	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	# 	thresh_nongaus = cv2.morphologyEx(thresh_nongaus, cv2.MORPH_OPEN, kernel)

	# 	#dilation_kernel = np.ones((half_param/130,half_param/130),np.uint8) 
	# 	#thresh_nongaus2 = cv2.erode(thresh_nongaus, dilation_kernel)
	# 	# find contours in the thresholded image, then initialize the
	# 	# digit contours lists
	# 	cnts = cv2.findContours(thresh_nongaus.copy(), cv2.RETR_TREE,
	# 		cv2.CHAIN_APPROX_SIMPLE)
	# 	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# 	digitCnts = []
	# 	cv2.imshow('thresh_nongaus',thresh_nongaus)
	# 	cv2.imwrite('./screen/thresh_nongaus_%d.png'%i,thresh_nongaus)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	# 	# loop over the digit area candidates
	# 	for c in cnts:
	# 		# compute the bounding box of the contour
	# 		(x, y, w, h) = cv2.boundingRect(c)
	# 		# if the contour is sufficiently large, it must be a digit
	# 		if (h >= 0.05 * image_h or w >= 0.05 * image_w) and ( h <= 0.5 * image_h or w <= 0.5 * image_w):
	# 			digitCnts.append(c)
	# 	print("len digitcnts", len(digitCnts))
		
	# 	# sort the contours from left-to-right, then initialize the
	# 	# actual digits themselves
	# 	# cv2.drawContours(output, digitCnts, -1, (0,255,0), 3)
	# 	# cv2.imshow('thresh_cnt',output)
	# 	# cv2.waitKey(0)
	# 	# cv2.destroyAllWindows()

	# 	digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
	# 	digits = []
	# 	rois = []
	# 	# loop over each of the digits
	# 	for c in digitCnts:
	# 		# extract the digit ROI
	# 		(x, y, w, h) = cv2.boundingRect(c)
	# 		rois.append( [x, y, w, h] )

	# 	rois = filter_rois(rois)
	# 	for [x, y, w, h] in rois:
	# 		roi = thresh_nongaus[y:y + h, x:x + w]
	# 		cv2.imwrite('./curr_pic/roi_%d.png'%i,roi)
	# 		############call your function here############
	# 		############use roi as input############
	# 		############return a digit as string named single_digit############
	# 		#result += single_digit

	# result = result[:3]+"\n" + result[3:] #for this machine result should have exactly 6 digits
	# return(result)

if __name__ == '__main__':
	main("area2.png", "sub2heat.jpg", [])