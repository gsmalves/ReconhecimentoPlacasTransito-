import numpy as np
import cv2
from classes.SegmentImage import SegmentImage
from pathlib import Path
import sys


path = Path(sys.path[0])
caminhoImagem = str(path.absolute()) + '/video1.mp4'

cap = cv2.VideoCapture(caminhoImagem)
#############################################
 # Armazena os frames por segundo do vídeo
# fps = cap.get(cv2.CAP_PROP_FPS)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Armazena as colunas do vídeo
# frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# # Armazena as linhas do vídeo
# frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
brightness = 180
threshold = 0.60         # PROBABILDADE THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# # SETUP DA CAMERA
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10, brightness)
# IMPORTANDO O ARQUIVO DE TREINAMENTO
# pickle_in=open('model_trained.p',"rb")  ## rb = READ BYTE
# model=pickle.load(pickle_in)
 
frame_counter=0
while True:
    frame_counter += 1
    if (frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    # LER A IMAGEM DA CAMERA
    success, imgOrignal = cap.read()
    scale_percent = 100
    width = int(imgOrignal.shape[1] * scale_percent / 100)
    height = int(imgOrignal.shape[0] * scale_percent / 100)
    tamanho = (width, height)

    imgOrignal = cv2.resize(imgOrignal, tamanho, cv2.INTER_LINEAR)
    if not success:
        break  
    img_copy =imgOrignal.copy()
    segmentImage = SegmentImage(imgOrignal)
    img, detections = segmentImage.segment()
    for i in detections:
        cv2.rectangle(img_copy, (i[0],i[1]),(i[0]+i[2],i[1]+i[2]),  (0,0,255), 2, 8, 0)
    cv2.imshow("Resultado", img_copy)

    if cv2.waitKey(int(fps)) and 0xFF == ord('q'):
        break
    
cap.release()