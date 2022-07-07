import numpy as np
import cv2
import pickle
import pandas as pd
from classes.SegmentImage import SegmentImage
from pathlib import Path
import sys


path = Path(sys.path[0])
caminhoImagem = str(path.absolute()) + '/video.mp4'

cap = cv2.VideoCapture(caminhoImagem)
#############################################
 # Armazena os frames por segundo do vídeo
# fps = cap.get(cv2.CAP_PROP_FPS)
fps = int(60/cap.get(cv2.CAP_PROP_FPS))

# Armazena as colunas do vídeo
# frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# # Armazena as linhas do vídeo
# frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
brightness = 180
threshold = 0.75         # PROBABILDADE THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
labelFile = 'labels.csv' # ARQUIVO COM NOMES DE CADA CLASSE
labelData=pd.read_csv(labelFile)

# # SETUP DA CAMERA
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10, brightness)
# IMPORTANDO O ARQUIVO DE TREINAMENTO
# pickle_in=open('model_trained.p',"rb")  ## rb = READ BYTE
# model=pickle.load(pickle_in)
filename = 'model_trained.p'
with open(filename, 'rb') as file:  
    model = pickle.load(file)

 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getClassName(classNo):
    classNames = np.loadtxt(fname=labelFile, delimiter=',', dtype='str', skiprows=1)
    #get class name from class no using labels.csv
    for classT in classNames:
        if classT[0]==str(classNo[0]):
            return classT[1]
    return ''
frame_counter=0
while True:
    frame_counter += 1
    if (frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    # LER A IMAGEM DA CAMERA
    success, imgOrignal = cap.read()
    scale_percent = 50
    width = int(imgOrignal.shape[1] * scale_percent / 100)
    height = int(imgOrignal.shape[0] * scale_percent / 100)
    tamanho = (width, height)

    imgOrignal = cv2.resize(imgOrignal, tamanho, cv2.INTER_LINEAR)
    if not success:
        break  
    
    segmentImage = SegmentImage(imgOrignal)
    # PROCESSa IMAGEM
    img = segmentImage.segment()
    # img = cv2.resize(img, (120, 120))
    # img = preprocessing(img)
    # cv2.imshow("Imagem processada", img)
    # img = img.reshape(1, 120, 120, 1)
    # cv2.putText(imgOrignal, "CLASSE: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(imgOrignal, "PROB.: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # # PREDIÇÃO DA IMAGEM
    # predictions = model.predict(img)
    # classIndex = np.argmax(model.predict(img), axis=-1)
    # probabilityValue =np.amax(predictions)
    # if probabilityValue > threshold:
    #     #print(getClassName(classIndex))
    #     cv2.putText(imgOrignal,str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    #     cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # elif probabilityValue < threshold/2:
    #     cv2.putText(imgOrignal,'', (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    #     cv2.putText(imgOrignal, '', (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Resultado", imgOrignal)

    if cv2.waitKey(int(fps)) and 0xFF == ord('q'):
        break
    
cap.release()