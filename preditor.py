
import cv2
from classes.SegmentImage import SegmentImage
from pathlib import Path
import sys


path = Path(sys.path[0])
caminhoImagem = str(path.absolute()) + '/video3.mp4'

# Carrega o vídeo
cap = cv2.VideoCapture(caminhoImagem)

# Armazena os frames por segundo do vídeo
# fps = cap.get(cv2.CAP_PROP_FPS)
fps = int(cap.get(cv2.CAP_PROP_FPS))


#Conta o número de frames do vídeo
frame_counter=0
while True:
    frame_counter += 1
    if (frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    # LER A IMAGEM DA CAMERA
    success, img_original = cap.read()
    scale_percent = 100
    width = int(img_original.shape[1] * scale_percent / 100)
    height = int(img_original.shape[0] * scale_percent / 100)
    tamanho = (width, height)
    # Redimensiona a imagem
    img_original = cv2.resize(img_original, tamanho, cv2.INTER_LINEAR)
    if not success:
        break  
    # Cria uma instância da classe SegmentImage passando a imagem original
    segmentImage = SegmentImage(img_original)
    img, detections = segmentImage.segment()
    # Mostra a iamgem das placas encontradas com um contorno vermelho
    for i in detections:
        cv2.rectangle(img_original, (i[0],i[1]),(i[0]+i[2],i[1]+i[2]),  (0,0,255), 2, 8, 0)
    cv2.imshow("Resultado", img_original)

    if cv2.waitKey(int(fps)) and 0xFF == ord('q'):
        break
    
cap.release()