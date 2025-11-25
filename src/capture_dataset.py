"""
Script para captura de gestos de mão e criação de dataset
Usa MediaPipe para detectar landmarks da mão e salva em CSV
"""

import cv2
import mediapipe as mp
import csv
import os
import time
from datetime import datetime

class GestureDatasetCapture:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
    def normalize_landmarks(self, landmarks):
        """Normaliza os landmarks da mão para valores relativos"""
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        z_coords = [lm.z for lm in landmarks]
        
        # Normaliza baseado no pulso (landmark 0)
        base_x, base_y, base_z = x_coords[0], y_coords[0], z_coords[0]
        
        normalized = []
        for i in range(21):  # 21 landmarks no MediaPipe
            normalized.extend([
                landmarks[i].x - base_x,
                landmarks[i].y - base_y,
                landmarks[i].z - base_z
            ])
        
        return normalized
    
    def capture_gesture(self, gesture_name, num_samples=100):
        """Captura samples de um gesto específico"""
        # Cria diretório datasets se não existir
        os.makedirs('datasets', exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro ao abrir a câmera!")
            return
        
        dataset_path = os.path.join('datasets', f'{gesture_name}.csv')
        samples_collected = 0
        countdown = 3
        start_time = time.time()
        capturing = False
        
        print(f"\n=== Capturando gesto: {gesture_name} ===")
        print(f"Objetivo: {num_samples} amostras")
        print("Pressione 'ESPAÇO' para iniciar a captura")
        print("Pressione 'Q' para sair")
        
        # Prepara arquivo CSV
        file_exists = os.path.exists(dataset_path)
        csv_file = open(dataset_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        
        # Escreve cabeçalho se arquivo não existir
        if not file_exists:
            header = ['gesture']
            for i in range(21):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            csv_writer.writerow(header)
        
        while cap.isOpened() and samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Desenha landmarks se detectados
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Captura dados se estiver em modo de captura
                    if capturing:
                        normalized = self.normalize_landmarks(hand_landmarks.landmark)
                        row = [gesture_name] + normalized
                        csv_writer.writerow(row)
                        samples_collected += 1
            
            # Interface
            if not capturing:
                cv2.putText(frame, f"Pressione ESPACO para iniciar", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Mostra progresso
                cv2.putText(frame, f"Capturando: {samples_collected}/{num_samples}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(frame, (10, 50), (10 + int(samples_collected/num_samples * 400), 70),
                            (0, 255, 0), -1)
            
            cv2.putText(frame, f"Gesto: {gesture_name}", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Captura de Dataset - Clash Royale Gestos', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not capturing:
                capturing = True
                print("Iniciando captura...")
        
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Captura concluída!")
        print(f"Total de amostras coletadas: {samples_collected}")
        print(f"Arquivo salvo em: {dataset_path}\n")


def main():
    capture = GestureDatasetCapture()
    
    print("="*60)
    print("SISTEMA DE CAPTURA DE DATASET - CLASH ROYALE GESTOS")
    print("="*60)
    print("\nGestos sugeridos:")
    print("1. thumbs_up (Polegar para cima)")
    print("2. peace (V de vitória)")
    print("3. fist (Punho fechado)")
    print("4. open_palm (Mão aberta)")
    print("5. pointing (Apontando)")
    print("6. ok_sign (Sinal de OK)")
    print()
    
    while True:
        gesture_name = input("Digite o nome do gesto (ou 'sair' para encerrar): ").strip()
        
        if gesture_name.lower() == 'sair':
            break
        
        if not gesture_name:
            print("Nome do gesto não pode ser vazio!")
            continue
        
        try:
            num_samples = int(input(f"Quantas amostras deseja capturar? (padrão: 100): ") or "100")
        except ValueError:
            num_samples = 100
        
        capture.capture_gesture(gesture_name, num_samples)
        
        continuar = input("\nDeseja capturar outro gesto? (s/n): ").strip().lower()
        if continuar != 's':
            break
    
    print("\n✓ Programa encerrado. Dataset salvo em ./datasets/")


if __name__ == "__main__":
    main()
