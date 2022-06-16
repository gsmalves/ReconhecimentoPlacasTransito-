import numpy as np
import cv2
import pickle
import pandas as pd

#############################################
 
frameWidth= 640         # RELOSUÇÃO DE IMAGENS
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABILDADE THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
labelFile = 'labels.csv' # ARQUIVO COM NOMES DE CADA CLASSE
labelData=pd.read_csv(labelFile)

# SETUP DA CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
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
        print( classT[0], classNo[0])
        if classT[0]==str(classNo[0]):
            return classT[1]
    return ''
while True:

    # LER A IMAGEM DA CAMERA
    success, imgOrignal = cap.read()
    
    # PROCESSa IMAGEM
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (120, 120))
    img = preprocessing(img)
    cv2.imshow("Imagem processada", img)
    img = img.reshape(1, 120, 120, 1)
    cv2.putText(imgOrignal, "CLASSE: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROB.: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDIÇÃO DA IMAGEM
    predictions = model.predict(img)
    classIndex = np.argmax(model.predict(img), axis=-1)
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
        #print(getClassName(classIndex))
        cv2.putText(imgOrignal,str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Resultado", imgOrignal)
    elif probabilityValue < threshold/2:
        cv2.putText(imgOrignal,'', (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, '', (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Resultado", imgOrignal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break