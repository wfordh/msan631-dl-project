from collections import deque
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms

def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1

cap = cv2.VideoCapture(0)
lower_green = np.array([110, 50, 50])
upper_green = np.array([130, 255, 255])
pts = deque(maxlen=512)
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
digit = np.zeros((200, 200, 3), dtype=np.uint8)
flag = 0
ans1 = ''
ans2 = ''
ans3 = ''

while (cap.isOpened()):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    res = cv2.bitwise_and(img, img, mask=mask)
    cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    center = None

    if len(cnts) >= 1:
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 200:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            pts.appendleft(center)
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
    elif len(cnts) == 0:
        if len(pts) != []:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
            if len(blackboard_cnts) >= 1:
                cnt = max(blackboard_cnts, key=cv2.contourArea)
                print(cv2.contourArea(cnt))
                if cv2.contourArea(cnt) > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    digit = blackboard_gray[y:y + h, x:x + w]
                    newImage = cv2.resize(digit, (input_dim, input_dim))
                    newImage = np.array(newImage)
                    newImage = newImage.flatten()
                    newImage = newImage.reshape(newImage.shape[0], 1)
                    ans1 = 1 #Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
                    ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
                    ans3 = Digit_Recognizer_DL.predict(d3, newImage)
        pts = deque(maxlen=512)
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", img)
    k = cv2.waitKey(10)
    if k == 27:
        break

# cap = cv2.VideoCapture(0)

# while (cap.isOpened()):
#     ret, img = cap.read()
#     img, contours, thresh = get_img_contour_thresh(img)
#     ans1 = ''
#     ans2 = ''
#     ans3 = ''
#     if len(contours) > 0:
#         contour = max(contours, key=cv2.contourArea)
#         if cv2.contourArea(contour) > 2500:
#             # print(predict(w_from_model,b_from_model,contour))
#             x, y, w, h = cv2.boundingRect(contour)
#             # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
#             newImage = thresh[y:y + h, x:x + w]
#             newImage = cv2.resize(newImage, (28, 28))
#             newImage = np.array(newImage)
#             newImage = newImage.flatten()
#             newImage = newImage.reshape(newImage.shape[0], 1)
#             ans1 = 1#Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
#             ans2 = 2#Digit_Recognizer_NN.predict_nn(d2, newImage)
#             ans3 = 3#Digit_Recognizer_DL.predict(d3, newImage)

#     x, y, w, h = 0, 0, 300, 300
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv2.imshow("Frame", img)
#     cv2.imshow("Contours", thresh)
#     k = cv2.waitKey(10)
#     if k == 27:
#         break
