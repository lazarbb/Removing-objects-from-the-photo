import cv2
import numpy

#--------------------------------------------------------------------------------------------
# Return list of classes from coco.name
def getClasses():
    with open('Object_Detection_Files/coco.names', 'rt') as file:
        # Save class names as list of strings 
        classes = file.read().rstrip('\n').split('\n')
    return classes

# Using Mobile Net SSD model
# Return image processing configuration
def getConfiguration():
    net = cv2.dnn_DetectionModel('Object_Detection_Files/frozen_inference_graph.pb',
                                 'Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    return net

# Detect objects on image and return image with bounding boxes
def detectObjects(imageP, net, classes):
    img = cv2.imread(imageP)
    displayImg = cv2.imread(imageP)

    # Resize image to 600x600
    img = cv2.resize(img, [600, 600])
    displayImg = cv2.resize(img, [600, 600])

    # Running detection and saving object Ids, confidences and bounding boxes
    classIDs, confidences, boxes = net.detect(img, confThreshold = 0.5)

    # Convert to lists
    boxes = list(boxes)
    confidences = list(map(float, confidences))

    # Using NMS (Non Maximum Suppression) to remove boxes with lower thresholds
    boxesToOutput = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Outputing bounding boxes witouth overlaping
    for i in boxesToOutput:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        # Drawing bounding boxes on the image from the positions saved in x, y
        cv2.rectangle(displayImg, (x, y), (x + w, y + h), color = (0, 0, 255))

        # Writing object name in the bounding box from the strings saved in classes
        # classIDs[i] is the id of the object in coco.names but indexes in classes start
        # from 0, so we have to subtract by 1
        cv2.putText(displayImg, str(i + 1), (x + 1, y + 20),
                    cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)
    return [displayImg, img, boxes, boxesToOutput[-1]]

# Using data from detectObjects, create a mask and perform inptainting
def deleteSelectedObjects(selectedObjects, boxes, img):
    # In the mask pixels outside the bounding box sre set to black
    mask = numpy.zeros_like(img)
    for i in selectedObjects:
        box = boxes[i - 1]
        x, y, w, h = box[0], box[1], box[2], box[3]

        # In the mask pixels inside the bounding box sre set to white
        mask[y:y + h + 1, x: x + w + 1] = 255
    
    # Convert mask to 8-bit single-channel image
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  

    # Using the original image and the mask perform inpainting to
    # remove selected objects
    output = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    
    return output
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
    