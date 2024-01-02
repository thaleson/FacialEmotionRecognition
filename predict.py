import streamlit as st
import numpy as np
import cv2
import imutils
from keras.models import model_from_json
from keras.optimizers import SGD
from scipy.ndimage import zoom
from tempfile import NamedTemporaryFile

# Constantes
EMOTIONS = ["Neutro", "Nervoso", "Esnobe",
            "Com Medo", "Triste", "Surpreso", "Feliz"]
FACE_MODEL_ARCHITECTURE_PATH = 'models/Face_model_architecture.json'
FACE_MODEL_WEIGHTS_PATH = 'models/Face_model_weights.h5'
HAARCASCADE_PATH = 'models/haarcascade_frontalface_default.xml'

# Carrega o Modelo de Deep Learning
model = model_from_json(open(FACE_MODEL_ARCHITECTURE_PATH).read())
model.load_weights(FACE_MODEL_WEIGHTS_PATH)

# Prepara o Modelo para Compilação
sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# Compila o modelo
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# Extrai as Features


def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = int(np.floor(offset_coefficients[0] * w))
    vertical_offset = int(np.floor(offset_coefficients[1] * h))

    extracted_face = gray[y + vertical_offset:y + h,
                          x + horizontal_offset:x - horizontal_offset + w]
    new_extracted_face = zoom(
        extracted_face, (48. / extracted_face.shape[0], 48. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face

# Detecta as Faces


def detect_face(frame):
    faceCascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
    return gray, detected_faces

# Função principal do Streamlit


def main():
    st.title("Detecção de Emoções Faciais em tempo real e em  videos ")

    # Upload do arquivo de vídeo
    video_file = st.file_uploader(
        "Selecione um arquivo de vídeo", type=["mp4", "avi", "mov"])

    if video_file:
        # Salvar o vídeo temporariamente
        temp_video = NamedTemporaryFile(delete=False)
        temp_video.write(video_file.read())

        # Abre o vídeo
        video_capture = cv2.VideoCapture(temp_video.name)

        # Configurações iniciais do Streamlit
        frame_out = st.empty()

        while True:
            # Captura frame-by-frame
            ret, frame = video_capture.read()

            # Se não obtivermos um frame, chegamos ao final do vídeo
            if not ret:
                st.write("Fim do vídeo.")
                break

            frame = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detecta as Faces
            gray, detected_faces = detect_face(frame)

            # Previsões
            for face in detected_faces:
                (x, y, w, h) = face
                if w > 100:
                    # Extrai as features
                    extracted_face = extract_face_features(
                        gray, face, (0.075, 0.05))  # (0.075, 0.05)

                    # Prevendo emoções
                    predictions = model.predict(
                        extracted_face.reshape(1, 48, 48, 1))
                    prediction_result = np.argmax(predictions)

                    # Adiciona a caixa delimitadora
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)

                    # Anota a imagem principal com uma etiqueta
                    expression_text = EMOTIONS[prediction_result]
                    cv2.putText(frame, expression_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)

            # Converte o frame para o formato de vídeo suportado pelo Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Exibe o vídeo no Streamlit
            frame_out.image(frame, channels="RGB")

        # Quando tudo estiver pronto, libera a captura
        video_capture.release()


if __name__ == '__main__':
    main()
