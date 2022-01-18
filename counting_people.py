import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import math
from shapely.geometry import Point, Polygon
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject

# Initialize the parameters
confThreshold = 0.6  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')

parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg";
modelWeights = "yolov3.weights";

# load our serialized model from disk
print("[INFO] loading model...")
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# initialize the video writer
writer = None
 
# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None
 
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
 
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalDown = 0
totalUp = 0

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class "person"
        if classIds[i] == 0:
            rects.append((left, top, left + width, top + height))
            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)
            #drawPred(confidences[i], left, top, left + width, top + height)
            counting(objects, confidences[i])
    
def drawPred(Id, conf, left, top, right, bottom):
    # Draw a bounding box.
    #cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
    label = 'ID%s:%s' % (Id, label)
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

def counting(objects, confidence):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global totalDown
    global totalUp

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
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
            print("ID: ", objectID)
            print("Direction=", direction)
            print("Confidence= ", confidence)
            to.centroids.append(centroid)
            coords = [(190, 460), (380, 320), (850, 400), (700, 565)]
            coord_t = [(380, 320), (350, 400), (810, 490), (850, 400)]
            coord_b = [(190, 460), (270, 435), (770, 540), (700, 565)]
            poly_t = Polygon(coord_t)
            poly_b = Polygon(coord_b)
            poly = Polygon(coords)
            point = Point(centroid)
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object

                if direction < 0 and poly_t.contains(point) and poly.contains(point):
                    totalUp += 1
                    to.counted = True
 
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and poly_b.contains(point) and poly.contains(point):
                    totalDown += 1
                    to.counted = True
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        drawPred(objectID, confidence, centroid[0] - 50, centroid[1] + 30, centroid[0] +30, centroid[1] -30 )
        #text = "ID{}".format(objectID)
        #cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        #    cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 2)
        #cv.putText(frame, label, (centroid[0] - 10, centroid[1] + 20),
        #    cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (255, 178, 50), -1)
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv.putText(frame, text, (10, frameHeight - ((i * 20) + 20)),
            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

# Process inputs
winName = 'People Counting and Tracking System'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"

if (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_hung.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    region = [[200, 450], [360, 340], [830, 420], [710, 555]]
    region = np.array(region, dtype=np.int32).reshape((-1, 1, 2))
    regiontop = [[380, 320], [350, 400], [810, 490], [850, 400]]
    regiontop = np.array(regiontop, dtype=np.int32).reshape((-1, 1, 2))
    regionbottom = [[190, 460], [270, 435], [770, 540], [700, 565]]
    regionbottom = np.array(regionbottom, dtype=np.int32).reshape((-1, 1, 2))
    cv.polylines(frame, [regiontop], True, (0, 255, 0), 2)
    cv.polylines(frame, [regionbottom], True, (0, 255, 0), 2)
    cv.polylines(frame, [region], True, (0, 0, 255), 2)
    #cv.line(frame, (190, 460), (700, 565), (0, 255, 0), 3)
    #cv.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)
    #cv.rectangle(frame, (0, frameHeight // 2 - 30), (frameWidth, frameHeight // 2 + 30), (0, 255, 0), 2)
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255))

    # Write the frame with the detection boxes

    vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)


# Usage:  python counting_people.py --video D:/ML/test.mp4