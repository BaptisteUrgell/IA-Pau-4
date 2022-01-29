from cv2 import threshold
from .mylib.centroidtracker import CentroidTracker
from .mylib.trackableobject import TrackableObject
# from imutils.video import VideoStream
from imutils.video import FPS
from .mylib.mailer import Mailer
from .mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import pandas as pd
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn


from .models.common import DetectMultiBackend
from .utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from .utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from .utils.plots import Annotator, colors, save_one_box
from .utils.torch_utils import select_device, time_sync

import streamlit as st


def det(model, img, CLASSES):
    # Load model
    imgsz = 640
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)
    img = torch.tensor(img, dtype=torch.int)
    img = img.permute(0,3,1,2)
    #img = torch.unsqueeze(img, dim=0)
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    print(img.shape)
    return non_max_suppression(model(img), 0.3 , 0.2, CLASSES, False, max_det=100)
		
t0 = time.time()

def prep(input_vid, skip_frames=4, output_vid="test_vid_own", generate_vid=True, demo=False, demo_container=None, video_raw=None, video_name=None):
	CLASSES = [0,1,2,3,5,6,7,8,9]
	threshold_detection = 0.25
	class_count = []
	for i in range(0,10):
		class_count.append(0)

	vs = cv2.VideoCapture(input_vid)
 
	if demo:
		vs = video_raw
    
	W = None
	H = None
	count = 0
	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = [0]
	empty=[]
	empty1=[]

	# start the frames per second throughput estimator
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)
	
	if generate_vid:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(output_vid, fourcc, 30,(W, H), True)

	model = DetectMultiBackend("/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/surfnet_v2/best.pt", device="cpu") 
	framepack = []
	# loop over frames from the video stream

	output = []
	
	frame = vs.read()
	frame = frame[1]# if args.get("input", False) else frame
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if generate_vid:
		writer = cv2.VideoWriter(f'/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/output/{video_name.split(".")[0]}.avi',
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          10, (640,384))

	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		
		frame = vs.read()[1]
		if frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = cv2.resize(frame, (640,384))
		
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % skip_frames == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []
			framepack.append(frame)
			detectionspack  = det(model,framepack, CLASSES)
			#detections  = det(model,frame, CLASSES)
			# loop over the detections
			#print(detections[0])
			framepack= []
			print(detectionspack)
			for detections in detectionspack:
				for i in np.arange(0, detections.shape[0]):
					# extract the confidence (i.e., probability) associated
					# with the prediction
					print(detections[i])
					confidence = detections[i, 4]
					
					# filter out weak detections by requiring a minimum
					# confidence
					if confidence > threshold_detection: #args["confidence"]
						
						# extract the index of the class label from the
						# detections list
						idx = int(detections[i, 5])
						# if the class label is not a person, ignore it
						#if CLASSES[idx] != "person":
						#	continue

						# compute the (x, y)-coordinates of the bounding box
						# for the object
						box = detections[i, 0:4] #* np.array([W,H, W, H])
						(startX, startY, endX, endY) = box.int()

						output.append((count, x[0], startX.item(), startY.item(), endX.item(), endY.item(), idx,confidence.item()))
						
						# construct a dlib rectangle object from the bounding
						# box coordinates and then start the dlib correlation
						# tracker
						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(startX, startY, endX, endY)

						tracker.start_track(frame, rect)

						# add the tracker to our list of trackers so we can
						# utilize it during skip frames
						trackers.append((tracker,idx))
						count += 1*(skip_frames)
			

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for (tracker, idx_labels) in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append(((startX, startY, endX, endY), idx_labels))


		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		(objects, idx_labels) = ct.update(rects)

		# loop over the tracked objects
		for ((objectID, centroid), idx_label) in zip(objects.items(),idx_labels.items()):
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:
					totalUp += 1
					empty.append(totalUp)
					to.counted = True
						
					x = []
					# compute the sum of total people inside
					x.append(len(empty))
					class_count[int(idx_label[1][0])] += 1


			print("Class count: ", class_count)
			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

		# construct a tuple of information we will be displaying on the
		info = [
		("Total", totalUp),
		("N", 0),
		("Status", status),
		]

		info2 = [
		("Total people inside", x),
		]
		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

		# show the output frame
		dict_class = {1: 'Insulating material',
        4: 'Drum',
        2: 'Bottle-shaped',
        3: 'Can-shaped',
        5: 'Other packaging',
        6: 'Tire',
        7: 'Fishing net / cord',
        8: 'Easily namable',
        9: 'Unclear',
        0: 'Sheet / tarp / plastic bag / fragment'}
		
		if generate_vid:
			writer.write(frame)
    
		if demo:
			with demo_container.container():
				st.image(frame)
				frame_trash_count = [str(dict_class[i]) + ' : ' + str(ct) for (i, ct) in enumerate(class_count)]
				st.write(f'Trash detected :\n')
				df1 = pd.DataFrame(
					np.reshape(np.array(class_count), (1,10)),
    				columns=(dict_class[i] for i in range(10)))
				df1.index = ['Count']
				# df2 = pd.DataFrame(
				# 	np.reshape(np.array(class_count[5:]), (1,5)),
    			# 	columns=(dict_class[i+5] for i in range(5)))
				# df2.index = ['Count']
				st.bar_chart(df1.T)
				#st.bar_chart(df2)

				# for txt in frame_trash_count:
				# 	st.write(txt) 
		else:
			cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		
		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	vs.release()
	writer.release()
	#cv2.destroyAllWindows()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	return output, class_count