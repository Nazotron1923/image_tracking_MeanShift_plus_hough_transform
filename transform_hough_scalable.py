import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import distance
from time import sleep
from collections import defaultdict
import math

roi_defined = False

# Good for the b/w test images used
MIN_CANNY_THRESHOLD = 10
MAX_CANNY_THRESHOLD = 50



scale = np.array([0.96, 0.98, 1, 1.02, 1.04])
# scale = np.array([1])
# ANG = np.array([-5, -3, 0, 3, 5])

# for different images
CCPT = [1, 25, 50, 75, 100, 125, 150]

# video_name = 'VOT-Sunshade'
# seuil = 85
# video_name = 'VOT-Woman'
# seuil = 100
# video_name = 'VOT-Ball'
# seuil = 90
# video_name = 'VOT-Car'
# seuil = 110
#
video_name = 'Antoine_Mug'
seuil = 55

cap = cv2.VideoCapture(video_name+'.mp4')


def draw_angled_rec(x0, y0, width, height, angle, img):
	print('angle ------------------>', angle)
	_angle = angle * math.pi / 180.0
	b = math.cos(_angle) * 0.5
	a = math.sin(_angle) * 0.5
	pt0 = (int(x0 - a * height - b * width),int(y0 + b * height - a * width))
	pt1 = (int(x0 + a * height - b * width),int(y0 - b * height - a * width))
	pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
	pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

	cv2.line(img, pt0, pt1, (255, 255, 255), 2)
	cv2.line(img, pt1, pt2, (255, 255, 255), 2)
	cv2.line(img, pt2, pt3, (255, 255, 255), 2)
	cv2.line(img, pt3, pt0, (255, 255, 255), 2)
	return img

def f_dst_weights(frame, x,y,w,h):
	# print(frame.shape)
	X, Y, _ = frame.shape
	mask = np.zeros((X, Y))
	mask += 0.1
	ww = min(x+int(1.5*w), X) - max(x-int(w/2), 0)
	hh = min(y+int(1.5*h), Y) - max(y-int(h/2), 0)

	template = np.indices((ww, hh))
	template[0] += max(x-int(w/2), 0)
	template[1] += max(y-int(h/2), 0)

	target = np.array([x+int(w/2),y+int(h/2)]).reshape(1,2)

	tmp = template.reshape(2, ww*hh)

	d = distance.cdist(tmp.T, target, 'euclidean')
	d_correct = d.reshape(ww,hh)

	std = 25
	m = 0

	b =  (1/(std*((2*np.pi)**0.5)))*np.exp( -((d_correct - m)**2)/(2*std*std) )
	b_min = np.min(b)
	cv2.normalize(b,b,0.3,1,cv2.NORM_MINMAX)
	mask[max(x-int(w/2), 0):min(x+int(1.5*w), X), max(y-int(h/2), 0): min(y+int(1.5*h), Y)] = b
	return mask

def get_gradient_magnitude(frame_g):
	dx = cv2.Sobel(frame_g,cv2.CV_64F,1,0,ksize=3)
	dy = cv2.Sobel(frame_g,cv2.CV_64F,0,1,ksize=3)
	# Compute the orientation of the image
	return  np.hypot(dx,dy).astype('uint8')


def get_gradient_orientation(frame_g):
	dx = cv2.Sobel(frame_g,cv2.CV_64F,1,0,ksize=3)
	dy = cv2.Sobel(frame_g,cv2.CV_64F,0,1,ksize=3)
	# Compute the orientation of the image
	return  (np.arctan2(dy,dx) * 180 / np.pi)



def build_r_table(obj):
	X,Y =  obj.shape
	gradient_magnitude = get_gradient_magnitude(obj)
	_ , filtered = cv2.threshold(gradient_magnitude, seuil, 255, cv2.THRESH_BINARY)
	cv2.imshow('r_table', filtered)
	orientation = get_gradient_orientation(filtered)
	orientation[filtered == 0] = -255
	unique_orientation = np.unique(orientation)

	r_table = dict()
	center = np.array([[int(X/2) ,int(Y/2)]])

	for teta in unique_orientation:
		if teta == -255:
			continue
		r_table[teta] = center - np.argwhere(orientation == teta)

	return r_table


def transform_hofe(image, r_table, x,y,w,h):

	# print('x: {}, y: {}, w: {}, h: {}'.format(x,y,w,h))

	X, Y = image.shape
	g = get_gradient_magnitude(image)
	_ , maskmask = cv2.threshold(g, seuil, 255, cv2.THRESH_BINARY)

	# cv2.imshow('get_gradient_magnitude', maskmask)
	teta = get_gradient_orientation(maskmask)
	teta[maskmask == 0] = -255

	max_v = - np.inf
	cur_v = - np.inf
	max_s = 1
	max_ang = 0
	center = np.array([w,h])

	# for aa in ANG:

	for s in scale:

		vot = np.zeros(image.shape)

		for t in r_table:
			tmp = np.argwhere(teta == t)
			if tmp.shape[0] == 0 :
				continue

			for r in r_table[t]:

				#  for angle transform
				# tr = np.array([r[0]*math.cos(aa) + r[1]*math.sin(aa),-r[0]*math.sin(aa) + r[1]*math.cos(aa)])
				# ind_for_vot = (tmp + s*tr).astype(int)

				# just scale transform
				ind_for_vot = (tmp + s*r).astype(int)
				ind_for_vot = ind_for_vot[ (ind_for_vot[:,0] < X) & (ind_for_vot[:,0] > 0) &(ind_for_vot[:,1] < Y) & (ind_for_vot[:,1] > 0)  ]
				vot[ind_for_vot[:,0], ind_for_vot[:,1]] += 1

		# vot[max(x-w, 0):min(x+2*w, X), max(y - h, 0):min(y+2*h, Y)] += 200
		cur_v = np.amax(vot)
		# print('---- Scale : {}, v: {}'.format(s, cur_v))
		if cur_v > max_v:
			max_v = cur_v
			max_s = s
			# max_ang  = aa
			centers = np.argwhere(vot == cur_v)
			center = centers.mean(axis = 0).astype('int')
			# center = centers[0]

	print('----------------- MAX Scale : {}, MAX Angle: {}'.format( max_s, max_ang))
	# if max_s != 1:
	# 	for r in r_table:
	# 		# tr = np.array([[r_table[r][:,0]*math.cos(max_ang) + r_table[r][:,1]*math.sin(max_ang)],[-r_table[r][:,0]*math.sin(max_ang) + r_table[r][:,1]*math.cos(max_ang)]])
	# 		# print('trtrtrtrtrtr------>',tr.shape)
	# 		# tr = tr.reshape(tr.shape[2], 2)
	# 		# chage r-table agter scale transformation
	# 		r_table[r] = (max_s*r_table[r])

	return center[0], center[1], int(w*max_s) , int(h*max_s) # max_ang



def define_ROI(event, x_p, y_p, flags, param):
	global x,y,w,h,roi_defined
	# if the left mouse button was clicked,
	# record the starting ROI coordinates
	if event == cv2.EVENT_LBUTTONDOWN:
		x, y = x_p, y_p
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		x2, y2 = x_p, y_p
		w = abs(x2-x)
		h = abs(y2-y)
		x = min(x,x2)
		y = min(y,y2)
		roi_defined = True

x,y,w,h = None, None, None, None
# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break

track_window = (x,y,w,h)

x_hofe, y_hofe, w_hofe, h_hofe = x,y,w,h
# set up the ROI for tracking
roi = frame[y:y+h+1, x:x+w+1]

# cv2.imshow('roi',roi)
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
grey_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


RT = build_r_table(grey_roi)
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))

# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
roi_hist_start = roi_hist.copy()
# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cpt = 1
while(1):
	# sleep(0.49)
	ret ,frame = cap.read()
	if ret == True:
		X, Y, _ = frame.shape
		frame_c = frame.copy()
		frame_cc = frame.copy()

		# frame_cc =  draw_angled_rec(100, 200, 50, 100, cpt, frame_cc)
		# cv2.imshow('frame_cc',frame_cc)


		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# Backproject the model histogram roi_hist onto the
		# current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
		dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

		m_dst = f_dst_weights(frame, y,x,h,w)
		tmp = dst*m_dst
		tmp = tmp.astype('uint8')

		# grey scale
		frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gradient_magnitude = get_gradient_magnitude(frame_g)
		_ , filtered = cv2.threshold(gradient_magnitude, seuil, 255, cv2.THRESH_BINARY)
		cv2.imshow('filtered_gradient_magnitude', filtered)



		y_hofe, x_hofe, h_hofe, w_hofe =  transform_hofe(frame_g, RT, y_hofe, x_hofe, h_hofe, w_hofe)
		# print('y_hofe, x_hofe, h_hofe, w_hofe')
		# print(y_hofe, x_hofe, h_hofe, w_hofe)
		# #
		#
		# frame_tracked_Hofe_t = cv2.rectangle(frame, (x_hofe - 1, y_hofe - 1), (x_hofe+1,y_hofe+1), (255,255,0) ,2)
		#

		grey_roi = frame_g[y_hofe - int(h_hofe/2):y_hofe+int(h_hofe/2)+1, x_hofe- int(w_hofe/2):x_hofe+int(w_hofe/2)+1]
		# grey_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		RT = build_r_table(grey_roi)

		frame_tracked_Hofe_t = cv2.rectangle(frame, (x_hofe - int(w_hofe/2), y_hofe-int(h_hofe/2)), (x_hofe+int(w_hofe/2),y_hofe+int(h_hofe/2)), (255,255,255) ,2)
		cv2.imshow('frame_tracked_Hofe_t', frame_tracked_Hofe_t)
		# frame_tracked_Hofe_ttt =  draw_angled_rec(x_hofe, y_hofe, w_hofe, h_hofe, angleangle, frame_cc)
		# cv2.imshow('frame_tracked_Hofe_ttt', frame_tracked_Hofe_ttt)



		# roi_t = frame_c[(y_hofe - int(h_hofe/2)):(y_hofe+int(h_hofe/2)), (x_hofe-int(w_hofe/2)):(x_hofe+int(w_hofe/2))]
		# grey_roi = cv2.cvtColor(roi_t, cv2.COLOR_BGR2GRAY)
		# RT = build_r_table(grey_roi)
		#

		# frame_tracked_Hofe = cv2.rectangle(frame, (10, 100), (30,160), (255,255,255) ,2)
		# cv2.imshow('frame_tracked_Hofe',frame_tracked_Hofe_t)
		# new_x, new_y,w_hofe,h_hofe = transform_hofe(frame_g, RT, y,x,w,h)
		# frame_tracked_Hofe = cv2.rectangle(frame, (new_y - int(w/2), new_x-int(h/2)), (new_y+int(w/2),new_x+int(h/2)), (255,255,255) ,2)
		# # frame_tracked_Hofe = cv2.rectangle(frame, (10, 100), (30,160), (255,255,255) ,2)
		# cv2.imshow('frame_tracked_Hofe',frame_tracked_Hofe)


		# apply meanshift to dst to get the new location
		ret, track_window = cv2.meanShift(tmp, track_window, term_crit)

		# Draw a blue rectangle on the current image
		x,y,w,h = track_window
		frame_tracked = cv2.rectangle(frame, (x, y), (x+w,y+h), (255,0,0) ,2)
		cv2.imshow('Sequence',frame_tracked)



		k = cv2.waitKey(60) & 0xff

		if k == 27:
		    	break
		elif k == ord('s'):
		    	cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)

		if cpt in CCPT:
			cv2.imwrite('h_52_Video_{}_Frame_{}.png'.format(video_name, cpt), frame_tracked)
			cv2.imwrite('h_52_Video_{}_filtered_Frame_{}.png'.format(video_name, cpt), filtered)
		cpt += 1
	else:
		break

cv2.destroyAllWindows()
cap.release()
