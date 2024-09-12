#import libraries
import cv2 as cv
import numpy as np
import math

cam = cv.VideoCapture(0)

while cam.isOpened():
  _, img = cam.read()
  cv.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0) 
  
  hand_region = img[100:300, 100:300] #cropping the image to get the hand region
  grey_scale = cv.cvtColor(hand_region, cv.COLOR_BGR2GRAY) #converting to gray scale
  blur = cv.GaussianBlur(grey_scale, (35, 35), 0) #blurring the image - smoothing of image
  _, thresh = cv.threshold(blur, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU) #converting back to binary image

  contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE) #finding contours
  count1 = max(contours, key=lambda x: cv.contourArea(x)) #finding largest contour
  x, y, w, h = cv.boundingRect(count1)
  cv.rectangle(hand_region, (x, y), (x + w, y + h), (0, 0, 255), 0) #drawing a bounding rectangle around the hand
  hull = cv.convexHull(count1) #finding convex hull - enclosing all the points in a structure
  drawing = np.zeros(hand_region.shape, np.uint8)
  cv.drawContours(drawing, [count1], 0, (0, 255, 0), 0) #drawing the largest contour - the hand
  cv.drawContours(drawing, [hull], 0, (0, 0, 255), 0) #drawing the hull
  hull = cv.convexHull(count1, returnPoints=False)
  defects = cv.convexityDefects(count1, hull) #finding convexity defects of the contour
  count_defects = 0
  cv.drawContours(thresh, contours, -1, (0, 255, 0), 3)

  for i in range(defects.shape[0]): # go thru each defect
      s, e, f, _ = defects[i, 0] #the diff points that define the convexity defect
      start = tuple(count1[s][0])
      end = tuple(count1[e][0])  
      far = tuple(count1[f][0])
      #side lengths
      a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
      b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
      c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)        
      #this a,b,c create a triangle
      angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
      
      if angle <= 90:
                count_defects += 1
                cv.circle(hand_region, far, 1, [0, 0, 255], -1)

      cv.line(hand_region, start, end, [0, 255, 0], 2)

  if count_defects == 0:
    cv.putText(img, "1 finger", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
  elif count_defects == 1:
    cv.putText(img, "2 fingers", (5, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
  elif count_defects == 2:
    cv.putText(img, "3 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
  elif count_defects == 3:
    cv.putText(img, "4 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
  elif count_defects == 5:
    cv.putText(img, "5 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))

  cv.imshow("Gesture", img)
  k = cv.waitKey(10) #waits for a key input to end, here waiting for 10 milliseconds
  if k == 27: #wait for 'esc' key
    break

cam.release()
cv.destroyAllWindows()

