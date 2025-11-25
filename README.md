# Projeto de Reconhecimento de Gestos - Clash Royale Emotes

Sistema de reconhecimento de gestos de mão em tempo real que exibe emotes do Clash Royale usando Python, MediaPipe e TensorFlow.

## 📋 Requisitos

- Python 3.12
- Webcam funcional
- Windows/Linux/Mac

## 🚀 Instalação

### 1. Clone ou baixe o projeto

```bash
cd "c:\Users\aluno\Desktop\Projeto IA"
```

### 2. Instale as dependências

```bash
C:/Users/aluno/AppData/Local/Programs/Python/Python313/python.exe -m pip install -r requirements.txt
```

## 📖 Como Usar

### Passo 1: Capturar Dataset

Execute o script de captura para criar seu dataset de gestos:

```bash
C:/Users/aluno/AppData/Local/Programs/Python/Python313/python.exe src/capture_dataset.py
```

**Instruções:**
1. Digite o nome do gesto (ex: thumbs_up, peace, fist, open_palm, pointing, ok_sign)
2. Defina quantas amostras deseja capturar (recomendado: 100-200 por gesto)
3. Pressione **ESPAÇO** para iniciar a captura
4. Faça o gesto em frente à câmera
5. O sistema capturará automaticamente os frames
6. Pressione **Q** para sair

**Dicas para captura:**
- Capture pelo menos 3-5 gestos diferentes
- Use iluminação adequada
- Varie ligeiramente a posição da mão
- Capture 100-200 amostras por gesto

### Passo 2: Treinar o Modelo

Após capturar os gestos, treine o modelo de IA:

```bash
C:/Users/aluno/AppData/Local/Programs/Python/Python313/python.exe src/train_model.py
```

O treinamento irá:
- Carregar todos os datasets capturados
- Criar uma rede neural
- Treinar o modelo (padrão: 50 épocas)
- Salvar o modelo treinado em `models/`

### Passo 3: Adicionar Emotes

1. Crie ou baixe imagens de emotes do Clash Royale (formato PNG com transparência preferível)
2. Renomeie as imagens com o **mesmo nome** dos gestos que você capturou
   - Exemplo: `thumbs_up.png`, `peace.png`, `fist.png`
3. Coloque as imagens na pasta `emotes/`

### Passo 4: Executar Reconhecimento

Execute a aplicação principal:

```bash
C:/Users/aluno/AppData/Local/Programs/Python/Python313/python.exe src/main.py
```

**Como funciona:**
- A webcam será ativada
- Faça um gesto em frente à câmera
- Se reconhecido, o emote correspondente aparecerá na tela por 2 segundos
- Pressione **Q** para sair

## 📁 Estrutura do Projeto

```
Projeto IA/
├── datasets/          # Dados de gestos capturados (CSV)
├── models/           # Modelos treinados
├── emotes/           # Imagens de emotes do Clash Royale
├── src/
│   ├── capture_dataset.py   # Script de captura de gestos
│   ├── train_model.py        # Script de treinamento
│   └── main.py               # Aplicação principal
├── requirements.txt  # Dependências do projeto
└── README.md        # Este arquivo
```

## 🎮 Gestos Sugeridos

Para uma experiência similar ao Clash Royale, recomendamos capturar estes gestos:

1. **thumbs_up** - Polegar para cima (👍)
2. **peace** - V de vitória (✌️)
3. **fist** - Punho fechado (✊)
4. **open_palm** - Mão aberta (✋)
5. **pointing** - Apontando (☝️)
6. **ok_sign** - Sinal de OK (👌)

## 🛠️ Tecnologias Utilizadas

- **OpenCV** - Captura e processamento de vídeo
- **MediaPipe** - Detecção de landmarks da mão
- **TensorFlow/Keras** - Treinamento de rede neural
- **NumPy** - Operações numéricas
- **Scikit-learn** - Pré-processamento de dados

## 🔧 Configurações Avançadas

### Ajustar Confiança de Detecção

No arquivo `src/main.py`, linha ~125:
```python
if confidence > 0.7:  # Ajuste este valor (0.0 a 1.0)
```

### Duração do Emote

No arquivo `src/main.py`, linha ~52:
```python
self.emote_duration = 2.0  # segundos
```

### Parâmetros de Treinamento

No arquivo `src/train_model.py`, você pode ajustar:
- Número de épocas
- Tamanho do batch
- Arquitetura da rede neural

## ❓ Solução de Problemas

### Erro: "Modelo não encontrado"
- Execute primeiro `capture_dataset.py` e depois `train_model.py`

### Baixa precisão no reconhecimento
- Capture mais amostras (200-300 por gesto)
- Melhore a iluminação durante a captura
- Treine por mais épocas
- Certifique-se de fazer gestos consistentes

### Câmera não funciona
- Verifique se a webcam está conectada
- Feche outros programas que possam estar usando a câmera

## 📝 Licença

Projeto educacional - Livre para uso e modificação.

## 🎓 Autor

Gustavo Henrique Bispo Costa
João Luiz Souza Pereira
