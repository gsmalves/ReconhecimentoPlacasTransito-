import cv2
import numpy as np


class SegmentImage:
    def __init__(self, image):
        self.image = image

    def segment(self):
        image = self.image
        
        
        #Segmentação da imagem
        low_red1 = np.array([0, 120, 70])
        high_red1 = np.array([10, 255, 255])
        low_red2 = np.array([170, 120, 70])
        high_red2 = np.array([180, 255, 255])

        #converte a imagem para HSV
        image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(image_HSV, low_red1, high_red1)
        mask2 = cv2.inRange(image_HSV, low_red2, high_red2)
   
        
        image_threshold = mask1 + mask2
        

        #Aplica o filtro de dilatação para melhorar a segmentação
        elem2 = cv2.getStructuringElement(
            cv2.MORPH_RECT, (5, 5))  # Elemento estruturante

        image_threshold = cv2.morphologyEx(image_threshold, cv2.MORPH_DILATE, elem2)
 
 
        # Preenchimento de buracos 
        res = cv2.findContours(
            image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        

        # Encontra os contornos
        contours = res[-2]  
        img_pl = np.zeros(image.shape[0:2])
        cv2.fillPoly(img_pl, pts=contours, color=255)
        houghCircles = np.zeros(image.shape[0:2])

        # Pega da imagem original os pontos com contornos vermelhos
        img_red = cv2.bitwise_and(image, image, mask=img_pl.astype(np.uint8))

        
        # Aplica o algoritmo de Hough para detectar os círculos
        circles = cv2.HoughCircles(cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY),cv2.HOUGH_GRADIENT,1,image.shape[0]/16,
        param1=100,param2=30, minRadius=10, maxRadius=200)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            #Desenha os círculos encontrados
            for i in circles[0,:]:
                cv2.circle(houghCircles,(i[0],i[1]),i[2] + 10,(255,255,255),-1)
        
        # Objetos circulares e com borda vermelha
        threshold_circle_red = cv2.bitwise_and(
        houghCircles.astype(np.uint8), img_pl.astype(np.uint8))

        #Encontrar os pontos com contornos vermelhos e circular
        res, _ = cv2.findContours(
            threshold_circle_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Interage com cada contorno.
        detections = []
        for i in range(len(res)):
            # Descobre o retângulo delimitador de cada contorno
            a = cv2.contourArea(res[i],False)
            if(a > 50):
                x,y,w,h = cv2.boundingRect(res[i])
                if w > h: 
                    max=w 
                else: 
                    max=h
                detections.append([x, y, max]);

        cv2.imshow('Imagem segmentada', threshold_circle_red)


        return threshold_circle_red, detections
