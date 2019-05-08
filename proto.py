import cv2
import numpy as np
import argparse
import math
parser = argparse.ArgumentParser(description='Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)')
parser.add_argument('--input', help='Path to input image.', required=True)
parser.add_argument('--model', default="frozen_east_text_detection.pb",
					help='Path to a binary .pb file of the model containing trained weights.')
parser.add_argument('--width', type=int, default=320,
					help='Preprocess input image by resizing to a specific width. It should be a multiple of 32.')
parser.add_argument('--height',type=int, default=320,
					help='Preprocess input image by resizing to a specific height. It should be a multiple of 32.')
parser.add_argument('--thr',type=float, default=0.5,
					help='Confidence threshold.')
parser.add_argument('--nms',type=float, default=0.4,
					help='Non-maximum suppression threshold.')
args = parser.parse_args()
def decode(scores, geometry, scoreThresh):
	detections = []
	confidences = []
	height = scores.shape[2]
	width = scores.shape[3]
	for y in range(height):
		scoresData = scores[0][0][y]
		x0_data = geometry[0][0][y]
		x1_data = geometry[0][1][y]
		x2_data = geometry[0][2][y]
		x3_data = geometry[0][3][y]
		anglesData = geometry[0][4][y]
		for x in range(0, width):
			score = scoresData[x]
			# If score is lower than threshold score, move to next x
			if(score < scoreThresh):
				continue

			# Calculate offset
			offsetX = x * 4.0
			offsetY = y * 4.0
			angle = anglesData[x]

			# Calculate cos and sin of angle
			cosA = math.cos(angle)
			sinA = math.sin(angle)
			h = x0_data[x] + x2_data[x]
			w = x1_data[x] + x3_data[x]

			# Calculate offset
			offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], 
						offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

			# Find points for rectangle
			p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
			p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
			center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
			detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
			confidences.append(float(score))

	# Return detections and confidences
	return [detections, confidences]
def detectAndCropText(img, confThreshold, nmsThreshold, inpWidth, inpHeight, net):
	#get height and width
	#reduce height to 1/3 of original
	height_ = int(img.shape[0]/3)
	width_ = img.shape[1]
	#crop to only top 1/3 of image
	img = img[:height_]

	#scaling factors for later resizing
	rW = width_/float(inpWidth)
	rH = height_/float(inpHeight)
def showImage(img):
	cv2.imshow("img", img)
	cv2.waitKey(0)
def main():
	#store arguements
	img_name = args.input
	confThreshold = args.thr
	nmsThreshold = args.nms
	inpWidth = args.width
	inpHeight = args.height
	model = args.model

	#load model
	net = cv2.dnn.readNet(model)

	outnames = ["feature_fusion/Conv_7/Sigmoid",
				"feature_fusion/concat_3"]
	#read image
	img = cv2.imread(img_name)

	#get height and width
	#reduce height to 1/3 of original
	height_ = int(img.shape[0]/3)
	width_ = img.shape[1]
	#crop to only top 1/3 of image
	img = img[:height_]

	#scaling factors for later resizing
	rW = width_/float(inpWidth)
	rH = height_/float(inpHeight)

	blob = cv2.dnn.blobFromImage(img, 1.0, (inpWidth, inpHeight), 
								(123.68, 116.78, 103.94), True, False)
	net.setInput(blob)
	#get scores and geometry
	scores, geometry = net.forward(outnames)

	#get boxes boundaries and confidences
	boxes, confidences = decode(scores, geometry, confThreshold)

	#apply Non-Max Supression
	indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)

	if len(indices) == 0:
		print("No text detected. Terminating.")
		return
	#find max dimensions of text
	min_x = width_
	min_y = height_
	max_x = 0
	max_y = 0
	for i in indices:
		vertices = cv2.boxPoints(boxes[i[0]])
		# scale the bounding box coordinates based on the respective ratios
		for j in range(4):
			vertices[j][0] *= rW
			vertices[j][1] *= rH
		for j in range(4):
			#find minimum and maximum bounds of text
			min_x = int(min(min_x, vertices[j][0]))
			min_y = int(min(min_y, vertices[j][1]))
			max_x = int(max(max_x, vertices[j][0]))
			max_y = int(max(max_y, vertices[j][1]))
	#add some padding so text isn't so tight
	min_x = max(0, min_x-10)
	min_y = max(0, min_y-5)
	max_y += 5
	max_x += 10
	img = img[min_y:max_y, min_x:max_x]
	showImage(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite("asdf.jpg", img)
if __name__ == '__main__':
	main()