from xmlrpc.client import boolean
from DetectAndRemove import *
import PySimpleGUI as pg

selectedObjects = []
img = []
boxes = []
numberOfObjects = 0
toggleDown = False
 
classes = getClasses()
net = getConfiguration()

# Theme
pg.theme('DarkAmber')

# Layout
layout = [
    [pg.Text('Pick an image'), pg.Input(key = '-IN-'), pg.FileBrowse(file_types = (('images', '*.png'),))
    , pg.Button('Load and detect')],
    [pg.Text('Select ids      '), pg.InputText(), pg.Button('Toggle visibility', disabled = True)],
]

# Window
window = pg.Window('Detect and Delete', layout)

# Event loop
while True:
    event, values = window.read()

    # Check if Load and detect is clicked
    if event == 'Load and detect':
        # Check if image was selected
        if(values['-IN-']):
            toggleDown = False
            window['Toggle visibility'].update(disabled = False)
            displayImg, img, boxes, numberOfObjects = detectObjects(values['-IN-'], net, classes)
            cv2.imshow('Object detection', displayImg)

    # Check if Toggle visibility is clicked
    elif event == 'Toggle visibility':
        # toggleDown is used to toggle between images
        # If False then perform object deletion and set toggleDown to True so
        # the next time it is clicked it will show the image witch object detection
        if toggleDown == False:
            toggleDown = True

            # Remove all elements that are not digits, digits less than 0 or higher than the number of detected objects
            # and covert to list of integers
            listOfIds = values[0].strip(',').replace(" ", "").split(',')
            listOfIds = [s for s in listOfIds if s.isdigit()]
            listOfIds = [int(i) for i in listOfIds if int(i) > 0 and int(i) <= numberOfObjects + 1]

            # Call object detection
            outputImage = deleteSelectedObjects(listOfIds, boxes, img)
            cv2.imshow('Object detection', outputImage)

        # If True then show the image witch object detection and set toggleDown to False so
        # the next time it is clicked it will remove selected objects
        elif toggleDown == True:
            toggleDown = False
            cv2.imshow('Object detection', displayImg)

    # Check if exit button is clicked
    elif event == pg.WIN_CLOSED:
        break

# Close the window
window.close()