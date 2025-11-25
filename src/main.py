"""
Aplicação principal - Reconhecimento de gestos em tempo real
Exibe emotes do Clash Royale quando detecta gestos específicos
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
from PIL import Image
import pygame


class GestureRecognizer:
    def __init__(self, model_path='models/gesture_model.keras',
                 encoder_path='models/label_encoder.pkl'):
        # Carrega modelo
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Execute train_model.py primeiro!")
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Modelo carregado de: {model_path}")
        
        # Carrega label encoder
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder não encontrado em {encoder_path}")
        
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"✓ Label encoder carregado: {list(self.label_encoder.classes_)}")
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Inicializa pygame mixer para áudio
        pygame.mixer.init()
        
        # Carrega emotes e sons
        self.emotes = self.load_emotes()
        self.sounds = self.load_sounds()
        
        # Controle de exibição
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.show_emote = False
        self.emote_timer = 0
        self.emote_duration = 2.0  # segundos
        self.last_played_gesture = None  # Evita tocar o mesmo som repetidamente
        
    def load_emotes(self, emotes_dir='emotes'):
        """Carrega imagens de emotes"""
        emotes = {}
        
        if not os.path.exists(emotes_dir):
            print(f"⚠ Diretório de emotes não encontrado: {emotes_dir}")
            print("  Crie a pasta 'emotes' e adicione imagens PNG dos emotes")
            return emotes
        
        for filename in os.listdir(emotes_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                gesture_name = os.path.splitext(filename)[0]
                img_path = os.path.join(emotes_dir, filename)
                
                # Carrega imagem com transparência
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    emotes[gesture_name] = img
                    print(f"  ✓ Emote carregado: {gesture_name}")
        
        return emotes
    
    def load_sounds(self, sounds_dir='sounds'):
        """Carrega arquivos de áudio dos emotes"""
        sounds = {}
        
        if not os.path.exists(sounds_dir):
            print(f"⚠ Diretório de sons não encontrado: {sounds_dir}")
            print("  Crie a pasta 'sounds' e adicione arquivos de áudio (.mp3, .wav, .ogg)")
            os.makedirs(sounds_dir, exist_ok=True)
            return sounds
        
        for filename in os.listdir(sounds_dir):
            if filename.endswith(('.mp3', '.wav', '.ogg')):
                gesture_name = os.path.splitext(filename)[0]
                sound_path = os.path.join(sounds_dir, filename)
                
                try:
                    sound = pygame.mixer.Sound(sound_path)
                    sounds[gesture_name] = sound
                    print(f"  ✓ Som carregado: {gesture_name}")
                except Exception as e:
                    print(f"  ✗ Erro ao carregar som {filename}: {e}")
        
        return sounds
    
    def play_sound(self, gesture_name):
        """Toca o som correspondente ao gesto"""
        if gesture_name in self.sounds:
            try:
                self.sounds[gesture_name].play()
            except Exception as e:
                print(f"Erro ao tocar som {gesture_name}: {e}")
    
    def normalize_landmarks(self, landmarks):
        """Normaliza landmarks da mão"""
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        z_coords = [lm.z for lm in landmarks]
        
        base_x, base_y, base_z = x_coords[0], y_coords[0], z_coords[0]
        
        normalized = []
        for i in range(21):
            normalized.extend([
                landmarks[i].x - base_x,
                landmarks[i].y - base_y,
                landmarks[i].z - base_z
            ])
        
        return np.array(normalized).reshape(1, -1)
    
    def overlay_transparent(self, background, overlay, x, y, scale=1.0):
        """Sobrepõe imagem com transparência"""
        if scale != 1.0:
            overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
        
        h, w = overlay.shape[:2]
        
        # Ajusta posição para não sair da tela
        if x + w > background.shape[1]:
            x = background.shape[1] - w
        if y + h > background.shape[0]:
            y = background.shape[0] - h
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        
        # Verifica se tem canal alpha
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                background[y:y+h, x:x+w, c] = (
                    alpha * overlay[:, :, c] +
                    (1 - alpha) * background[y:y+h, x:x+w, c]
                )
        else:
            background[y:y+h, x:x+w] = overlay
        
        return background
    
    def run(self):
        """Executa reconhecimento em tempo real"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro ao abrir a câmera!")
            return
        
        print("\n" + "="*60)
        print("RECONHECIMENTO DE GESTOS - CLASH ROYALE")
        print("="*60)
        print("Gestos reconhecidos:", list(self.label_encoder.classes_))
        print("\nPressione 'Q' para sair")
        print("="*60 + "\n")
        
        import time
        last_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Detecta mão
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Desenha landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Prediz gesto
                    features = self.normalize_landmarks(hand_landmarks.landmark)
                    prediction = self.model.predict(features, verbose=0)
                    gesture_idx = np.argmax(prediction)
                    confidence = prediction[0][gesture_idx]
                    
                    if confidence > 0.7:  # Threshold de confiança
                        self.current_gesture = self.label_encoder.inverse_transform([gesture_idx])[0]
                        self.gesture_confidence = confidence
                        
                        # Ativa exibição de emote
                        if not self.show_emote:
                            self.show_emote = True
                            self.emote_timer = time.time()
                            
                            # Toca som se disponível e não foi o último tocado
                            if self.current_gesture != self.last_played_gesture:
                                self.play_sound(self.current_gesture)
                                self.last_played_gesture = self.current_gesture
            
            # Gerencia exibição de emote
            if self.show_emote:
                elapsed = time.time() - self.emote_timer
                if elapsed < self.emote_duration:
                    # Exibe emote
                    if self.current_gesture in self.emotes:
                        emote_img = self.emotes[self.current_gesture]
                        # Centraliza emote
                        emote_x = w // 2 - 100
                        emote_y = h // 2 - 100
                        frame = self.overlay_transparent(frame, emote_img, emote_x, emote_y, scale=0.5)
                else:
                    self.show_emote = False
                    self.last_played_gesture = None  # Reseta para permitir tocar novamente
            
            # Interface
            # Fundo para texto
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            
            if self.current_gesture:
                cv2.putText(frame, f"Gesto: {self.current_gesture}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confianca: {self.gesture_confidence*100:.1f}%", (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Nenhum gesto detectado", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, "Pressione 'Q' para sair", (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Clash Royale - Reconhecimento de Gestos', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        recognizer = GestureRecognizer()
        recognizer.run()
    except FileNotFoundError as e:
        print(f"\nErro: {e}")
        print("\nPara usar este sistema:")
        print("1. Capture dataset: python src/capture_dataset.py")
        print("2. Treine o modelo: python src/train_model.py")
        print("3. Execute este script: python src/main.py")
    except Exception as e:
        print(f"\nErro inesperado: {e}")


if __name__ == "__main__":
    main()
