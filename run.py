#!/usr/bin/env python

import argparse
import cv2
from numpy import empty, nan
import os
import sys
import time

import CMT
import numpy as np
import util


CMTobj = CMT.CMT()

parser = argparse.ArgumentParser(description='Track an object.')

parser.add_argument('inputpath', nargs='?', help='The input path.')
parser.add_argument('--challenge', dest='challenge', action='store_true', help='Enter challenge mode.')
parser.add_argument('--preview', dest='preview', action='store_const', const=True, default=None, help='Force preview')
parser.add_argument('--no-preview', dest='preview', action='store_const', const=False, default=None, help='Disable preview')
parser.add_argument('--no-scale', dest='estimate_scale', action='store_false', help='Disable scale estimation')
parser.add_argument('--with-rotation', dest='estimate_rotation', action='store_true', help='Enable rotation estimation')
parser.add_argument('--bbox', dest='bbox', help='Specify initial bounding box.')
parser.add_argument('--pause', dest='pause', action='store_true', help='Pause after every frame and wait for any key.')
parser.add_argument('--output-dir', dest='output', help='Specify a directory for output data.')
parser.add_argument('--quiet', dest='quiet', action='store_true', help='Do not show graphical output (Useful in combination with --output-dir ).')
parser.add_argument('--skip', dest='skip', action='store', default=None, help='Skip the first n frames', type=int)

args = parser.parse_args()

CMTobj.estimate_scale = args.estimate_scale
CMTobj.estimate_rotation = args.estimate_rotation

if args.pause:
	pause_time = 0
else:
	pause_time = 10

if args.output is not None:
	if not os.path.exists(args.output):
		os.mkdir(args.output)
	elif not os.path.isdir(args.output):
		raise Exception(args.output + ' exists, but is not a directory')

if args.challenge:
	with open('images.txt') as f:
		images = [line.strip() for line in f]

	init_region = np.genfromtxt('region.txt', delimiter=',')
	num_frames = len(images)

	results = empty((num_frames, 4))
	results[:] = nan

	results[0, :] = init_region

	frame = 0

	im0 = cv2.imread(images[frame])
	im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
	im_draw = np.copy(im0)

	tl, br = (util.array_to_int_tuple(init_region[:2]), util.array_to_int_tuple(init_region[:2] + init_region[2:4] - 1))

	try:
		CMTobj.initialise(im_gray0, tl, br)
		while frame < num_frames:
			im = cv2.imread(images[frame])
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			CMTobj.process_frame(im_gray)
			results[frame, :] = CMTobj.bb

			# Advance frame number
			frame += 1
	except:
		pass  # Swallow errors

	np.savetxt('output.txt', results, delimiter=',')

else:
	# Clean up
	cv2.destroyAllWindows()

	preview = args.preview

	if args.inputpath is not None:

		# If a path to a file was given, assume it is a single video file
		if os.path.isfile(args.inputpath):
			cap = cv2.VideoCapture(args.inputpath)

			#Skip first frames if required
			if args.skip is not None:
				cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, args.skip)


		# Otherwise assume it is a format string for reading images
		else:
			cap = util.FileVideoCapture(args.inputpath)

			#Skip first frames if required
			if args.skip is not None:
				cap.frame = 1 + args.skip

		# By default do not show preview in both cases
		if preview is None:
			preview = False

	else:
		# If no input path was specified, open camera device
		cap = cv2.VideoCapture(0)
		if preview is None:
			preview = True

	# Check if videocapture is working
	if not cap.isOpened():
		print 'Unable to open video input.'
		sys.exit(1)

	while preview:
		status, im = cap.read()
		cv2.imshow('Preview', im)
		k = cv2.waitKey(10)
		if not k == -1:
			break

	# Read first frame
	status, im0 = cap.read()
	im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
	im_draw = np.copy(im0)

	if args.bbox is not None:
		# Try to disassemble user specified bounding box
		values = args.bbox.split(',')
		try:
			values = [int(v) for v in values]
		except:
			raise Exception('Unable to parse bounding box')
		if len(values) != 4:
			raise Exception('Bounding box must have exactly 4 elements')
		bbox = np.array(values)

		# Convert to point representation, adding singleton dimension
		bbox = util.bb2pts(bbox[None, :])

		# Squeeze
		bbox = bbox[0, :]

		tl = bbox[:2]
		br = bbox[2:4]
	else:
		# Get rectangle input from user
		(tl, br) = util.get_rect(im_draw)

	print 'using', tl, br, 'as init bb', im_gray0.shape


	CMTobj.initialise(im_gray0, tl, br)

	cmt_list = []
	cmt_list.append(CMTobj)

	active_points_ini = CMTobj.active_keypoints.shape[0]
	print "Active_points_ini=", active_points_ini
	frame = 1
	last_inserted_frame = 0	
	tl_min_x = im_gray0.shape[1]
	tl_min_y = im_gray0.shape[0]
	br_max_x = 0
	br_max_y = 0
	while True:
		# Read image
		status, im = cap.read()
		if not status:
			break
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im_draw = np.copy(im)

		cmt_idx = 0
		max_keypoints = 0
		max_keypoint_inst = None
		tl_min_x = im_gray0.shape[1]
		tl_min_y = im_gray0.shape[0]
		br_max_x = 0
		br_max_y = 0
		
		for CMT_inst in cmt_list:
			tic = time.time()
			CMT_inst.process_frame(im_gray)
			toc = time.time()
	
			# Display results
	
			# Draw updated estimate
			if CMT_inst.has_result:
	
				cv2.line(im_draw, CMT_inst.tl, CMT_inst.tr, (255, (cmt_idx % 6 ) * 50, 0), 4)
				cv2.line(im_draw, CMT_inst.tr, CMT_inst.br, (255, (cmt_idx % 6 ) * 50, 0), 4)
				cv2.line(im_draw, CMT_inst.br, CMT_inst.bl, (255, (cmt_idx % 6 ) * 50, 0), 4)
				cv2.line(im_draw, CMT_inst.bl, CMT_inst.tl, (255, (cmt_idx % 6 ) * 50, 0), 4)
				
				if CMT_inst.tl[0] < tl_min_x:
					tl_min_x = CMT_inst.tl[0]
					 
				if CMT_inst.tl[1] < tl_min_y:
					tl_min_y = CMT_inst.tl[1] 

				if CMT_inst.br[0] > br_max_x:
					br_max_x = CMT_inst.br[0] 

				if CMT_inst.br[1] > br_max_y:
					br_max_y = CMT_inst.br[1] 

				if CMT_inst.tr[0] > br_max_x:
					br_max_x = CMT_inst.tr[0] 

				if CMT_inst.tr[1] < tl_min_y:
					tl_min_y = CMT_inst.tr[1] 


				if CMT_inst.bl[0] < tl_min_x:
					tl_min_x = CMT_inst.bl[0] 

				if CMT_inst.bl[1] > br_max_y:
					br_max_y = CMT_inst.bl[1] 

				print tl_min_x, tl_min_y,  br_max_x, br_max_y


				if CMT_inst.active_keypoints.shape[0] > max_keypoints:
					max_keypoints = CMT_inst.active_keypoints.shape[0]
					max_keypoint_inst = CMT_inst
	
			util.draw_keypoints(CMT_inst.tracked_keypoints, im_draw, (255, 255, 255))
			# this is from simplescale
			util.draw_keypoints(CMT_inst.votes[:, :2], im_draw)  # blue
			util.draw_keypoints(CMT_inst.outliers[:, :2], im_draw, (0, 0, 255))
	
			if args.output is not None:
				# Original image
				cv2.imwrite('{0}/input_{1:08d}.png'.format(args.output, frame), im)
				# Output image
				cv2.imwrite('{0}/output_{1:08d}.png'.format(args.output, frame), im_draw)
	
				# Keypoints
				with open('{0}/keypoints_{1:08d}.csv'.format(args.output, frame), 'w') as f:
					f.write('x y\n')
					np.savetxt(f, CMT_inst.tracked_keypoints[:, :2], fmt='%.2f')
	
				# Outlier
				with open('{0}/outliers_{1:08d}.csv'.format(args.output, frame), 'w') as f:
					f.write('x y\n')
					np.savetxt(f, CMT_inst.outliers, fmt='%.2f')
	
				# Votes
				with open('{0}/votes_{1:08d}.csv'.format(args.output, frame), 'w') as f:
					f.write('x y\n')
					np.savetxt(f, CMT_inst.votes, fmt='%.2f')
	
				# Bounding box
				with open('{0}/bbox_{1:08d}.csv'.format(args.output, frame), 'w') as f:
					f.write('x y\n')
					# Duplicate entry tl is not a mistake, as it is used as a drawing instruction
					np.savetxt(f, np.array((CMT_inst.tl, CMT_inst.tr, CMT_inst.br, CMT_inst.bl, CMT_inst.tl)), fmt='%.2f')
	
			if 0 and not args.quiet:
				cv2.imshow('main', im_draw)
	
				# Check key input
				k = cv2.waitKey(pause_time)
				key = chr(k & 255)
				if key == 'q':
					break
				if key == 'd':
					import ipdb; ipdb.set_trace()
	
	
	
			print '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(CMT_inst.center[0], CMT_inst.center[1], CMT_inst.scale_estimate, CMT_inst.active_keypoints.shape[0], 1000 * (toc - tic), frame), CMT_inst.rotation_estimate, cmt_idx
			cmt_idx += 1

		#if br_max_x > 0 and br_max_y > 0:
		if not args.quiet:
			cv2.line(im_draw, (tl_min_x, tl_min_y), (br_max_x, tl_min_y), (0,0, 255), 4)
			cv2.line(im_draw, (br_max_x, tl_min_y), (br_max_x, br_max_y), (0,0, 255), 4)
			cv2.line(im_draw, (br_max_x, br_max_y), (tl_min_x, br_max_y), (0,0, 255), 4)
			cv2.line(im_draw, (tl_min_x, br_max_y), (tl_min_x, tl_min_y), (0,0, 255), 4)
			
			cv2.imshow('main', im_draw)

			# Check key input
			k = cv2.waitKey(pause_time)
			key = chr(k & 255)
			if key == 'q':
				break
			if key == 'd':
				import ipdb; ipdb.set_trace()

		cmt_list2 = []
		for CMT_inst in cmt_list:
			if CMT_inst.active_keypoints.shape[0] > 1 and  CMT_inst.active_keypoints.shape[0] < ( active_points_ini * 1.5) :
				cmt_list2.append(CMT_inst)
		cmt_list = cmt_list2
		
		print max_keypoints,(active_points_ini / 2)
		if (max_keypoints > 1) and (max_keypoints < (active_points_ini / 1.5)):
			#print frame, (last_inserted_frame + 10)
			if frame > (last_inserted_frame + 5):
				print "Launching new cmt"
				CMTnew = CMT.CMT()
				CMTnew.estimate_scale = args.estimate_scale
				CMTnew.estimate_rotation = args.estimate_rotation
				#CMTnew.initialise(im_gray, max_keypoint_inst.tl, max_keypoint_inst.br)
				CMTnew.initialise(im_gray, (tl_min_x,tl_min_y), (br_max_x,br_max_y)         )
				cmt_list.append(CMTnew)
				last_inserted_frame = frame

		# Remember image
		im_prev = im_gray

		# Advance frame number
		frame += 1
