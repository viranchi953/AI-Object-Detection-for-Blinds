import numpy as np
import time
import cv2
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)
engine.setProperty('volume',0.9)

LABELS = open("coco.names").read().strip().split("\n")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)
frame_count = 0
start = time.time()
first = True
frames = []

while True:
	frame_count += 1

	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	frames.append(frame)
	if ret:
		key = cv2.waitKey(1)
		if frame_count % 60 == 0:
			end = time.time()

			(H, W) = frame.shape[:2]

			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			boxes = []
			confidences = []
			classIDs = []
			centers = []

			for output in layerOutputs:

				for detection in output:

					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					if confidence > 0.8:

						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
						centers.append((centerX, centerY))

			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

			texts = []

			if len(idxs) > 0:

				for i in idxs.flatten():

					centerX= centers[i][0]
					if centerX<=320:
						if(centerX>=288):
							W_pos = " In Front"
						elif(centerX>=160):
							W_pos = " At slight left"
						else:
							W_pos = " At Left"
					else:
						if(centerX<=352):
							W_pos = " In Front"
						elif(centerX<=480):
							W_pos = " At slight right"
						else:
							W_pos = " At right"

					texts.append(LABELS[classIDs[i]]+W_pos)
					print(texts)
			if texts:
				engine.say(texts);
				engine.runAndWait()
		cv2.imshow("Frame",frame)


cap.release()
cv2.destroyAllWindows()