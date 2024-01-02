# Script para preparar conjuntos de dados, carregar dados e gerar imagens em escala 
import pandas as pd
import numpy as np

# Crie um diretório e mantenha o arquivo fer2013.csv no diretório. 
# Dataset original em: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
data = pd.read_csv('data/fer2013.csv')
data = data['pixels']
data = [ dat.split() for dat in data]
data = np.array(data)
data = data.astype('float64')
data = [[np.divide(d,255.0) for d in dat] for dat in data]

# Salva os dados em formato numpy
np.save('data/Scaled.bin.npy',data)
