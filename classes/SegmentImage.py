import cv2
import sys
from pathlib import Path
from cv2 import waitKey
from cv2 import HoughCircles
from cv2 import threshold
import numpy as np


class SegmentImage:
    def __init__(self, image):
        self.image = image

    def segment(self):
        image = self.image
        
        low_red1 = np.array([0, 120, 70])
        high_red1 = np.array([10, 255, 255])
        low_red2 = np.array([170, 120, 70])
        high_red2 = np.array([180, 255, 255])
        sensitivity = 15
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([255,sensitivity,255])
        image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(image_HSV, low_red1, high_red1)
        mask2 = cv2.inRange(image_HSV, low_red2, high_red2)
        # mask3 = cv2.inRange(image_HSV, lower_white, upper_white)
        
        
        image_threshold = mask1 + mask2
        

        elem2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (10, 10))  # Elemento estruturante

        image_threshold = cv2.morphologyEx(image_threshold, cv2.MORPH_DILATE, elem2)
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 5
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.9

        params.filterByConvexity = True
        params.minConvexity = 0.2
        detector = cv2.SimpleBlobDetector_create(params)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(image_gray)

        blobs = np.zeros(image.shape)
        for x in range(1,len(keypoints)):
            blobs=cv2.circle(blobs, (np.int(keypoints[x].pt[0]),np.int(keypoints[x].pt[1])), radius=np.int(keypoints[x].size), color=(255), thickness=-1)

 
 
        # Preenchimento
        res = cv2.findContours(
            image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours = res[-2]  # for cv2 v3 and v4+ compatibility
        img_pl = np.zeros(image.shape[0:2])
        cv2.fillPoly(img_pl, pts=contours, color=255)

        blobs_ = cv2.inRange(blobs,255,255)
        
        # Objetos circulares e com bordar vermelha
        threshold_circle_red = cv2.bitwise_and(
        blobs_, img_pl.astype(np.uint8))

        result = np.zeros(threshold_circle_red.shape)
        res = res[0] if len(res) == 2 else res[1]
        for cnt in res:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) > 8:
                cv2.drawContours(result, [cnt], 0, (255, 255, 255), -1)
                
        
        result = cv2.bitwise_and(image, image, mask=threshold_circle_red.astype(np.uint8))
        
        res, _ = cv2.findContours(
            threshold_circle_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Interage com cada contorno.
        for i in range(len(res)):
            # Descobre o retângulo delimitador de cada contorno
            a = cv2.contourArea(res[i],False)
            if(a > 0):
                bounding_rect = cv2.boundingRect(res[i])
                cv2.rectangle(image, bounding_rect,  (0,0,255), 2, 8, 0)

        cv2.imshow('Imagem segmentada', threshold_circle_red)
        cv2.imshow('Filter HSV', img_pl)
        cv2.imshow('Filter HoughCircles', blobs)


        return result
