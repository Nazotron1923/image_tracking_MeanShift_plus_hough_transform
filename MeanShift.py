import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import distance
from time import sleep

roi_defined = False

CCPT = [1, 50, 100, 150, 200, 250, 300]
# video_name = 'VOT-Sunshade'
video_name = 'VOT-Woman'
# video_name = 'VOT-Ball'
# video_name = 'VOT-Car'
# video_name = 'Antoine_Mug'

cap = cv2.VideoCapture(video_name+'.mp4')


def f_dst_weights(frame, x,y,w,h):
	X, Y, _ = frame.shape
	weights = np.zeros((X, Y)) + 0.15

	# defining a zone of curiosity
	ww = min(x+int(1.5*w), X) - max(x-int(w/2), 0)
	hh = min(y+int(1.5*h), Y) - max(y-int(h/2), 0)
	template = np.indices((ww, hh))
	template[0] += max(x-int(w/2), 0)
	template[1] += max(y-int(h/2), 0)

	# the center from which the distance will be counted
	target = np.array([[x+int(w/2),y+int(h/2)]])

	# Calculate the distance from the center to all points of interest.
	d = distance.cdist(template.reshape(2, ww*hh).T, target, 'euclidean').reshape(ww,hh)
	# we use the Gaussian distribution to transform the distance
	std = 25
	gaussian = (1/(std*((2*np.pi)**0.5)))*np.exp( -((d)**2)/(2*std*std) )
	# normalization
	cv2.normalize(gaussian, gaussian, 0.15, 1, cv2.NORM_MINMAX)
	# Creating weights for the density.
	weights[max(x-int(w/2), 0):min(x+int(1.5*w), X), max(y-int(h/2), 0): min(y+int(1.5*h), Y)] = gaussian
	# cv2.imshow('Weights', weights)
	return weights



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
# set up the ROI for tracking
roi = frame[y:y+h+1, x:x+w+1]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<30, V<20 or V>235 are ignored
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# mask = cv2.inRange(hsv_roi, np.array((15.,30.,55.)), np.array((180.,235.,235.)))

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
	# sleep(0.2)
	print('cpt: ', cpt)
	ret ,frame = cap.read()
	if ret == True:
		X, Y, _ = frame.shape
		frame_c = frame.copy()
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# Backproject the model histogram roi_hist onto the
		# current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
		dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
		dst_weights = f_dst_weights(frame, y,x,h,w)


		weighted = dst*dst_weights
		weighted = weighted.astype('uint8')
		cv2.normalize(weighted,weighted,0,255,cv2.NORM_MINMAX)


		cv2.imshow('dst', dst)
		cv2.imshow('weighted dst',weighted)

		# apply meanshift to dst to get the new location
		ret, track_window = cv2.meanShift(weighted, track_window, term_crit)
		# Draw a blue rectangle on the current image
		x,y,w,h = track_window
		frame_tracked = cv2.rectangle(frame, (x, y), (x+w,y+h), (255,0,0) ,2)
		cv2.imshow('Sequence',frame_tracked)

		roi_current = frame_c[y:y+h+1, x:x+w+1]
		hsv_roi_current =  cv2.cvtColor(roi_current, cv2.COLOR_BGR2HSV)
		mask_current = cv2.inRange(hsv_roi_current, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
		roi_hist_current = cv2.calcHist([hsv_roi_current],[0],mask_current,[180],[0,180])
		cv2.normalize(roi_hist_current,roi_hist_current,0,255,cv2.NORM_MINMAX)

		model_difference = np.sum( (roi_hist -  roi_hist_current)**2 )/w*h
		print('Test {} :  {}'.format(w*h, model_difference))

		# if model_difference > 60000 :
		# 	roi_hist = roi_hist_current

		k = cv2.waitKey(60) & 0xff

		if k == 27:
		    	break
		elif k == ord('s'):
			cv2.imwrite('Video_{}_Frame_{}.png'.format(video_name, cpt), frame_tracked)
			cv2.imwrite('Video_{}_dst_Frame_{}.png'.format(video_name, cpt), dst)
			cv2.imwrite('Video_{}_weighted_Frame_{}.png'.format(video_name, cpt), weighted)

		# if cpt in CCPT:
		# 	cv2.imwrite('mod_Video_{}_Frame_{}.png'.format(video_name, cpt), frame_tracked)
		# 	cv2.imwrite('mod_Video_{}_dst_Frame_{}.png'.format(video_name, cpt), dst)
		# 	cv2.imwrite('mod_Video_{}_weighted_Frame_{}.png'.format(video_name, cpt), weighted)
		cpt += 1
	else:
		break

cv2.destroyAllWindows()
cap.release()
