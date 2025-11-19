"""
Script para treinar modelo de classificação de gestos
Usa TensorFlow/Keras para criar uma rede neural
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle


class GestureModelTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, dataset_dir='datasets'):
        """Carrega todos os arquivos CSV do diretório de datasets"""
        all_data = []
        
        if not os.path.exists(dataset_dir):
            print(f"Erro: Diretório {dataset_dir} não encontrado!")
            return None, None
        
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"Erro: Nenhum arquivo CSV encontrado em {dataset_dir}!")
            return None, None
        
        print(f"Carregando datasets...")
        for csv_file in csv_files:
            file_path = os.path.join(dataset_dir, csv_file)
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"  ✓ {csv_file}: {len(df)} amostras")
        
        # Combina todos os datasets
        full_dataset = pd.concat(all_data, ignore_index=True)
        
        # Separa features e labels
        X = full_dataset.drop('gesture', axis=1).values
        y = full_dataset['gesture'].values
        
        print(f"\nDataset total: {len(full_dataset)} amostras")
        print(f"Gestos únicos: {len(np.unique(y))}")
        print(f"Classes: {np.unique(y)}")
        
        return X, y
    
    def create_model(self, input_shape, num_classes):
        """Cria modelo de rede neural"""
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Treina o modelo"""
        # Codifica labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Divide dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=validation_split, random_state=42, stratify=y_encoded
        )
        
        print(f"\nDados de treino: {len(X_train)} amostras")
        print(f"Dados de teste: {len(X_test)} amostras")
        
        # Cria modelo
        self.model = self.create_model(X.shape[1], len(self.label_encoder.classes_))
        
        print("\nArquitetura do modelo:")
        self.model.summary()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Treina
        print(f"\nIniciando treinamento por {epochs} épocas...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Avalia
        print("\n" + "="*60)
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Acurácia no conjunto de teste: {test_accuracy*100:.2f}%")
        print(f"Loss no conjunto de teste: {test_loss:.4f}")
        print("="*60)
        
        return history
    
    def save_model(self, model_path='models/gesture_model.keras', 
                   encoder_path='models/label_encoder.pkl'):
        """Salva o modelo e o encoder"""
        os.makedirs('models', exist_ok=True)
        
        # Salva modelo
        self.model.save(model_path)
        print(f"\n✓ Modelo salvo em: {model_path}")
        
        # Salva label encoder
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Label encoder salvo em: {encoder_path}")


def main():
    print("="*60)
    print("TREINAMENTO DE MODELO - CLASH ROYALE GESTOS")
    print("="*60)
    
    trainer = GestureModelTrainer()
    
    # Carrega dataset
    X, y = trainer.load_dataset('datasets')
    
    if X is None or y is None:
        print("\nErro ao carregar dataset. Certifique-se de ter capturado dados primeiro!")
        print("Execute: python src/capture_dataset.py")
        return
    
    # Parâmetros de treinamento
    print("\n--- Configuração de Treinamento ---")
    try:
        epochs = int(input("Número de épocas (padrão: 50): ") or "50")
        batch_size = int(input("Tamanho do batch (padrão: 32): ") or "32")
    except ValueError:
        epochs = 50
        batch_size = 32
    
    # Treina modelo
    history = trainer.train(X, y, epochs=epochs, batch_size=batch_size)
    
    # Salva modelo
    trainer.save_model()
    
    print("\n✓ Treinamento concluído com sucesso!")
    print("\nPróximos passos:")
    print("1. Execute: python src/main.py")
    print("2. Adicione imagens de emotes na pasta ./emotes/")


if __name__ == "__main__":
    main()
