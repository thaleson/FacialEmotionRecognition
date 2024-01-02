# 😊 Detecção de Emoções Faciais em Vídeo 😊

Este é um projeto de detecção de emoções faciais em vídeos usando Deep Learning. Ele usa um modelo treinado para identificar as emoções (Neutro, Nervoso, Esnobe, Com Medo, Triste, Surpreso, Feliz) em faces detectadas em um vídeo.

## Estrutura do Projeto 📂

- **`models/`**: Contém os arquivos do modelo de Deep Learning.
  - `Face_model_architecture.json`: Arquivo JSON que descreve a arquitetura do modelo.
  - `Face_model_weights.h5`: Pesos do modelo treinado.

- **`videos/`**: Pasta contendo vídeos para testes.
  - `video1.mp4`, `video2.mp4`, ...: Vídeos de exemplo para testar a detecção de emoções faciais.

- **`utils/`**: Módulos utilitários do projeto.
  - `imutils.py`: Funções auxiliares para manipulação de imagens e vídeos.
  - Outros módulos utilitários, se aplicável.

- **`requirements.txt`**: Lista de dependências do projeto.

- **`predict.py`**: Script principal para executar a detecção de emoções em vídeos.

## Pré-requisitos 🛠️

Certifique-se de ter o Python e o pip instalados em seu sistema. Recomenda-se o uso de um ambiente virtual para isolar as dependências do projeto.

```bash
# Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv

# Ative o ambiente virtual
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

## Como Usar 🚀

``` Faça o clone do repositório:

git clone https://github.com/thaleson/FacialEmotionRecognition.git
cd seu-repositorio
```

## instale as dependências:

```pip install -r requirements.txt```

## Execute o script principal:
```python predict.py```


## Compatibilidade ✨
O projeto foi testado nos seguintes sistemas operacionais:

- Windows
- Linux
- macOS

## Contribuindo 🤝
Sinta-se à vontade para abrir problemas (issues) ou enviar solicitações de pull (pull requests) para melhorar este projeto.

## Licença 📝
Este projeto está licenciado sob a Sua Licença.
