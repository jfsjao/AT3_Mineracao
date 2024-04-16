#Baixar o database do link que esta no readme, descompactar o database e por fim deixar na pasta raiz, junto ao index.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
data_df = pd.read_csv('data.csv')
labels_df = pd.read_csv('labels.csv')

# Remover coluna de identificação e normalizar os dados
data_numeric = data_df.drop(columns=data_df.columns[0])
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_numeric)

# Aplicar PCA para reduzir a dimensionalidade
pca = PCA(n_components=50)  # Reduz para 50 componentes
data_pca = pca.fit_transform(data_normalized)

# Aplicar modelos de clustering
# K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
labels_kmeans = kmeans.fit_predict(data_pca)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
labels_hierarchical = hierarchical.fit_predict(data_pca)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(data_pca)

# Função para plotar os resultados do clustering
def plot_clusters(data, labels, title):
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(data[labels == label, 0], data[labels == label, 1], label=f'Cluster {label}')
    plt.title(title)
    plt.legend()
    plt.show()

# Plotar resultados
plot_clusters(data_pca, labels_kmeans, 'K-Means Clustering')
plot_clusters(data_pca, labels_hierarchical, 'Hierarchical Clustering')
plot_clusters(data_pca, labels_dbscan, 'DBSCAN Clustering')

# Imprimir quantos clusters foram gerados e os centróides para K-Means
print("Número de clusters em K-Means:", np.unique(labels_kmeans).size)
print("Centróides de K-Means:\n", kmeans.cluster_centers_)
print("Número de clusters em Hierarchical:", np.unique(labels_hierarchical).size)
print("Número de clusters em DBSCAN:", np.unique(labels_dbscan).size)
