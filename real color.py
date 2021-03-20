import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

while 1:
    _, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red color
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)

    # Blue color
    blue_lower = np.array([110,50,50], np.uint8)
    blue_upper = np.array([130,255,255], np.uint8)

    # yellow color
    yellow_lower = np.array([22, 60, 200], np.uint8)
    yellow_upper = np.array([60, 255, 255], np.uint8)

    # white color
    white_lower = np.array([0, 0, 200], np.uint8)
    white_upper = np.array([180, 20, 255], np.uint8)

    # black color
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 30], np.uint8)

    # green color
    green_lower = np.array([50, 50, 50], np.uint8)
    green_upper = np.array([70, 255, 255], np.uint8)

    # All color together
    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white = cv2.inRange(hsv, white_lower, white_upper)
    black = cv2.inRange(hsv, black_lower, black_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)

    # Morphological Transform,Dilation

    kernel = np.ones((10, 10), "uint8")

    red = cv2.dilate(red, kernel)
    res_red = cv2.bitwise_and(img, img, mask=red)

    blue = cv2.dilate(blue, kernel)
    res_blue = cv2.bitwise_and(img, img, mask=blue)

    yellow = cv2.dilate(yellow, kernel)
    res_yellow = cv2.bitwise_and(img, img, mask=yellow)

    white = cv2.dilate(white, kernel)
    res_white = cv2.bitwise_and(img, img, mask=white)

    black = cv2.dilate(black, kernel)
    res_black = cv2.bitwise_and(img, img, mask=black)

    green = cv2.dilate(green, kernel)
    res_green = cv2.bitwise_and(img, img, mask=green)

    # Tracking red
    contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

    # Tracking blue
    contours, hierarchy = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0),1)

    # Tracking yellow
    contours, hierarchy = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (70, 255, 255), 2)
            cv2.putText(img, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (70, 255, 255))

    # Tracking white
    contours, hierarchy = cv2.findContours(white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img, "White Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255))

    # Tracking black
    contours, hierarchy = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(img, "Black Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))

    # Tracking green
    contours, hierarchy = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

    cv2.imshow("Color Tracking", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
