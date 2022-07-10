import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
 
 
################# Parametros#####################
 
path = "dataset" # pasta com todos as imagens do dataset
labelFile = 'labels.csv' # arquivo com os nomes dos arquivos de imagens
batch_size_val=50  # quantos processar juntos
epochs_val=10
imageDimesions = (120,120,3)
testRatio = 0.2    #se 1000 forem divididas , 200 serão para teste 
validationRatio = 0.2 # 200 das restantes serão para a validação
###################################################
 
 
###############################    Importando as imagens 
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total de classes detectadas:",len(myList))
noOfClasses=len(myList)
print("Importando Classes.....")
for x in range (0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end =" ")
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)
 
############################### Separação de dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
 
# X_train = Matriz de imagens de treino
# y_train = ID DA CLASSE CORRESPONDENTE
 
############################### PARA VERIFICAR SE O NÚMERO DE IMAGENS CORRESPONDE AO NÚMERO DE LABELS PARA CADA CONJUNTO DE DADOS
print("Formas de dados")
print("Treinamento",end = "");print(X_train.shape,y_train.shape)
print("Validação",end = "");print(X_validation.shape,y_validation.shape)
print("Teste",end = "");print(X_test.shape,y_test.shape)
assert(X_train.shape[0]==y_train.shape[0]), "O numero de imagens de treino não corresponde ao numero de labels"
assert(X_validation.shape[0]==y_validation.shape[0]), "O numero de imagens de validação não corresponde ao numero de labels"
assert(X_test.shape[0]==y_test.shape[0]), "O numero de imagens de teste não corresponde ao numero de labels"
assert(X_train.shape[1:]==(imageDimesions))," A dimensão das imagens não corresponde ao padrão"
assert(X_validation.shape[1:]==(imageDimesions))," A dimensão das imagens não corresponde ao padrão"
assert(X_test.shape[1:]==(imageDimesions))," A dimensão das imagens não corresponde ao padrão"
 
 
############################### LER O ARQUIVO CSV
data=pd.read_csv(labelFile)
print("data shape ",data.shape,type(data))
###############################EXIBIR ALGUMAS AMOSTRAS DE IMAGENS DE TODAS  CLASSES
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j,row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)- 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j)+ "-"+row["Name"])
            num_of_samples.append(len(x_selected))
 
 
###############################EXIBIR UM GRÁFICO DE BARRAS MOSTRANDO O NÚMERO DE AMOSTRAS PARA CADA CATEGORIA
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribuição de classes")
plt.xlabel("Numero da classe")
plt.ylabel("Numero de imagens")
plt.show()
 
############################### Pre-processamento das imagens
 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)     # CONVERTE PARA ESCALA DE CINZA
    img = equalize(img)      # Equaliza a imagem
    img = img/255            # PARA NORMALIZAR VALORES ENTRE 0 E 1 EM VEZ DE 0 A 255
    return img
 
X_train=np.array(list(map(preprocessing,X_train)))  # Para cada imagem da matriz X_train, aplica a função preprocessing
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))
cv2.imshow("Imagens com escala de cinza",X_train[random.randint(0,len(X_train)-1)]) # TO CHECK IF THE TRAINING IS DONE PROPERLY
 
############################### ADD A PROFUNDIDE DE 1
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
 
 
############################### AUMENTA A DIMENSÃO DA IMAGEM
dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10)  # DEGREES
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)  # Gerador de dados para gerar imagens  do tamanho de batch_size
X_batch,y_batch = next(batches)
 
# Para mostrar as imagens aumentadas
fig,axs=plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()
 
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0],imageDimesions[1]))
    axs[i].axis('off')
plt.show()
 
 
y_train = to_categorical(y_train,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
 
############################### Rede neural convolucional
def myModel():
    no_Of_Filters=60
    size_of_Filter=(5,5) # Kernel que se move na imagem para achar as caracteristicas
    size_of_Filter2=(3,3)
    size_of_pool=(2,2)  # REDUZIR TODO O MAPA DE RECURSOS PARA GERNALIZAR MAIS, PARA REDUZIR O OVERFITTING
    no_Of_Nodes = 250   # NÚMERO DE NÓS NA CAMADA INTERMEDIÁRIA
    model= Sequential()
    model.add((Conv2D(no_Of_Filters,size_of_Filter,input_shape=(imageDimesions[0],imageDimesions[1],1),activation='relu')))#ADICIONAR MAIS CAMADAS DE CONVOLUÇÃO = MENOS RECURSOS, MAS PODE CAUSAR AUMENTO DA PRECISÃO
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
 
    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2,activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
 
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes,activation='relu'))
    model.add(Dropout(0.5)) # NÓS DE ENTRADA DEPENDEM DE OUTROS NÓS DE SAIDA
    model.add(Dense(noOfClasses,activation='softmax')) # NÓ DE SAIDA
    # COMPILE MODEL
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
 
 
############################### TRAIN
model = myModel()
print(model.summary())
history=model.fit(dataGen.flow(X_train,y_train,batch_size=batch_size_val),epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=1)
 
############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['treinamento','validacao'])
plt.title('loss')
plt.xlabel('epoca')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['treinamento','validacao'])
plt.title('Acurracia')
plt.xlabel('epoca')
plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Acuracia:',score[1])
 
 
# GUARDE O MODELO COMO OBJETO DE PICKLE
pickle_out= open("model_trained.p","wb")  # wb = WRITE BYTE
pickle.dump(model,pickle_out)
pickle_out.close()
cv2.waitKey(0)