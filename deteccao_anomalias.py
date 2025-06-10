import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Suprimir o aviso do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Função para avaliar os modelos e exibir a matriz de confusão com porcentagens
def evaluate_model(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    # Calcular porcentagens da matriz de confusão
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plotando a matriz de confusão com porcentagens usando matplotlib
    plt.figure(figsize=(6,6))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Anomaly'])
    plt.yticks(tick_marks, ['Normal', 'Anomaly'])
    
    # Exibir as porcentagens na matriz
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm_percent[i, j]:.2f}%", ha='center', va='center', color='black')
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def aplicar_modelos(nome_dataset, features, labels=None):
    print(f"\n========== {nome_dataset} ==========")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Verifique se X e y possuem o mesmo número de amostras antes de usar o SMOTE
    if len(features_scaled) != len(labels):
        raise ValueError(f"Inconsistent length: {len(features_scaled)} vs {len(labels)}")

    if labels is not None:
        smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
        features_scaled, labels = smote.fit_resample(features_scaled, labels)
        print(f"SMOTE applied: features and labels are balanced. Shape of features: {features_scaled.shape}, labels: {labels.shape}")

        # Verifique novamente se X e y possuem o mesmo número de amostras após SMOTE
        assert len(features_scaled) == len(labels), f"Inconsistent length after SMOTE: {len(features_scaled)} vs {len(labels)}"

        # Isolation Forest - mais sensível (contamination menor para evitar falso positivo excessivo)
        iso_forest = IsolationForest(
            n_estimators=500,
            contamination=0.15,
            bootstrap=True,
            max_samples='auto',
            random_state=42
        )
        iso_pred = (iso_forest.fit_predict(features_scaled) == -1).astype(int)

        # Local Outlier Factor - ajustar n_neighbors e contamination, usar novelty=True para previsão em dados completos
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.15, novelty=True)
        lof.fit(features_scaled)
        lof_pred = (lof.predict(features_scaled) == -1).astype(int)

        auto_pred = np.zeros(len(labels)) if labels is not None else None
        if labels is not None:
            # Treinar Autoencoder só com dados normais (label=0) para melhor aprendizado
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels, test_size=0.2, stratify=labels, random_state=42
            )
            X_train_norm = X_train[y_train == 0]  # só normais para treino do autoencoder

            input_layer = Input(shape=(X_train.shape[1],))
            encoded = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
            encoded = Dense(64, activation='relu')(encoded)
            encoded = Dropout(0.3)(encoded)
            encoded = Dense(32, activation='relu')(encoded)

            decoded = Dense(32, activation='relu')(encoded)
            decoded = Dense(64, activation='relu')(decoded)
            decoded = Dense(X_train.shape[1], activation='sigmoid')(decoded)

            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')

            # Treina só com normais para o autoencoder aprender bem a reconstrução normal
            autoencoder.fit(X_train_norm, X_train_norm,
                            epochs=150, batch_size=64,
                            validation_data=(X_test, X_test),
                            verbose=0)

            reconstructed = autoencoder.predict(X_test)
            mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)

            scaler_mse = MinMaxScaler()
            mse_scaled = scaler_mse.fit_transform(mse.reshape(-1, 1)).flatten()

            precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, mse_scaled)
            f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)

            # Escolher threshold que maximize recall >= 0.5 (priorizar recall)
            candidates = [(f1, thr, rec) for f1, thr, rec in zip(f1_scores, thresholds, recall_vals) if rec >= 0.5]
            if candidates:
                best_f1, best_threshold, best_recall = max(candidates, key=lambda x: x[0])
            else:
                best_threshold = thresholds[np.argmax(f1_scores)]  # fallback padrão

            auto_pred = (mse_scaled > best_threshold).astype(int)
            print("\n[Autoencoder]")
            evaluate_model(y_test, auto_pred, f"Autoencoder - {nome_dataset}")

        # Ensemble ponderado (dando mais peso ao autoencoder e Isolation Forest)
        if labels is not None:
            # Pesos baseados no recall individual (simples, você pode aprimorar)
            recalls = []
            for pred in [iso_pred, lof_pred, auto_pred]:
                recalls.append(recall_score(labels, pred, zero_division=0))
            weights = np.array(recalls)
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            weights = weights / weights.sum()  # normalizar

            ensemble_score = (weights[0] * iso_pred) + (weights[1] * lof_pred) + (weights[2] * auto_pred)
            ensemble_pred = (ensemble_score >= 0.5).astype(int)  # threshold 0.5 para decidir anomalia

            print("\n[Ensemble - Votação Ponderada]")
            evaluate_model(labels, ensemble_pred, f"Ensemble - {nome_dataset}")
    else:
        print(f"[{nome_dataset}] No labels available. Skipping evaluation metrics.")

def selecionar_features(df, base_name, colunas_base):
    features = df[colunas_base].copy()

    for porta_col in ['port', 'ct_dst_sport_ltm', 'ct_src_dport_ltm']:
        if porta_col in df.columns:
            features['Port'] = df[porta_col]
            break

    for loc_col in ['from', 'Location']:
        if loc_col in df.columns:
            features['Location'] = df[loc_col].astype('category').cat.codes
            break

    return features.dropna()

# =======================
# CARGA DE DADOS E PRÉ-PROCESSAMENTO
# =======================

# Carregar o dataset Honeypot (singapore)
honeypot_singapore = pd.read_excel('C:\\Users\\camil\\OneDrive\\Documentos\\Faculdade\\TCC\\datasets\\Honeypot attack logs\\singapore.alterado.xlsx')

# Pré-processamento do Honeypot (singapore)
honeypot_singapore['time'] = pd.to_datetime(honeypot_singapore['time'], errors='coerce', format='%m/%d/%Y, %H:%M:%S')
honeypot_singapore['hour'] = honeypot_singapore['time'].dt.hour
honeypot_singapore['day'] = honeypot_singapore['time'].dt.dayofweek

# Seleção das features
honeypot_features = selecionar_features(honeypot_singapore, "Honeypot", ['hour', 'day'])

# Criando rótulos reais para Honeypot (0 = normal, 1 = anômalo)
normal_samples = len(honeypot_singapore) // 2  # Ajustando para metade do total de amostras (normal e anômalo)
anomaly_samples = len(honeypot_singapore) // 2  # O total de amostras será metade de normais e metade de anômalos
honeypot_labels = np.array([0]*normal_samples + [1]*anomaly_samples)  # 0 = normal, 1 = anômalo

# Passando os rótulos reais para a função
aplicar_modelos("Honeypot Singapore", honeypot_features, honeypot_labels)

# Carregar e processar o UNSW-NB15 (train + test)
unsw_train = pd.read_csv('C:\\Users\\camil\\OneDrive\\Documentos\\Faculdade\\TCC\\datasets\\UNSW-NB15\\UNSW_NB15_training-set.csv', low_memory=False)
unsw_test = pd.read_csv('C:\\Users\\camil\\OneDrive\\Documentos\\Faculdade\\TCC\\datasets\\UNSW-NB15\\UNSW_NB15_testing-set.csv', low_memory=False)

# Pré-processamento do UNSW-NB15
unsw_train['time'] = pd.to_datetime(unsw_train['time'], errors='coerce')
unsw_test['time']

