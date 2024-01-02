# Imports
import numpy as np
import cv2

def translate(image, x, y):
	# Define a matriz de tradução e realiza a tradução
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Retorna a imagem traduzida
	return shifted

def rotate(image, angle, center = None, scale = 1.0):
	# Obtém as dimensões da imagem
	(h, w) = image.shape[:2]

	# Se o centro for None, inicialize-o como o centro da imagem
	if center is None:
		center = (w / 2, h / 2)

	# Executa a rotação
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Retorna a imagem girada
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# Inicializa as dimensões da imagem a ser redimensionada e pegue o tamanho da imagem
	dim = None
	(h, w) = image.shape[:2]

	# Se tanto a largura quanto a altura são None, então retorna a imagem original
	if width is None and height is None:
		return image

	# Verifica se a largura é None
	if width is None:
		# Calcula a proporção da altura e constrói as dimensões
		r = height / float(h)
		dim = (int(w * r), height)

	# Caso contrário, a altura é None
	else:
		# Calcula a proporção da largura e constrói as dimensões
		r = width / float(w)
		dim = (width, int(h * r))

	# Redimensiona a imagem
	resized = cv2.resize(image, dim, interpolation = inter)

	# Devolve a imagem redimensionada
	return resized

	