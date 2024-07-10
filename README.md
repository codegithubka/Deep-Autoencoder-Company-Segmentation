# Autoencoder-Based Clustering Analysis

This project focuses on using an autoencoder to learn a compact representation of high-dimensional data, which is then used for clustering analysis. The goal is to find the best autoencoder architecture and hyperparameters that improve the clustering performance, evaluated by metrics such as the silhouette score and the Davies-Bouldin index.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Autoencoder Architecture](#autoencoder-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Clustering Analysis](#clustering-analysis)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Dependencies](#dependencies)
- [Usage](#usage)

## Introduction

Autoencoders are neural networks used for learning efficient codings of input data. This project utilizes an autoencoder to encode a high-dimensional dataset into a lower-dimensional latent space. The encoded data is then clustered using various clustering algorithms, and the performance is evaluated to find the optimal autoencoder architecture and clustering parameters.

## Dataset

The dataset used in this project is synthetic and consists of various involvement columns and industry columns. The dataset file should be named `synthetic_data.csv` and should be placed in the `data/` directory.

## Preprocessing

The data is preprocessed by mapping categorical values to numerical weights and standardizing the features. The following steps are performed:

1. Load the dataset.
2. Map categorical values to numerical weights.
3. Assign weights to features.
4. Standardize the data.

## Autoencoder Architecture

The autoencoder architecture consists of an encoder and a decoder. The encoder compresses the input data into a lower-dimensional representation, while the decoder reconstructs the input data from this representation. The architecture is fine-tuned by adjusting the number of layers, the number of neurons per layer, dropout rates, and regularization techniques.

## Hyperparameter Tuning

A grid search is performed to find the best hyperparameters for the autoencoder. The following hyperparameters are tuned:

- Number of layers and neurons per layer.
- Encoding dimension.
- Dropout rate.
- L1 and L2 regularization.
- Batch size.
- Number of epochs.

## Clustering Analysis

The encoded data is clustered using the K-Means and Agglomerative Clustering algorithms. The optimal number of clusters is determined by evaluating the silhouette score for different cluster numbers.

## Evaluation

The clustering performance is evaluated using the following metrics:

- Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters.
- Davies-Bouldin Index: Measures the average similarity ratio of each cluster with its most similar cluster.

## Results

The best autoencoder architecture and hyperparameters are selected based on the highest silhouette score. The best parameters are printed along with their corresponding silhouette score.

## Visualization

The clustering results are visualized using dimensionality reduction techniques such as PCA, t-SNE, and UMAP. These visualizations help in understanding the distribution of clusters in the latent space.

## Dependencies

The project requires the following Python libraries:

- pandas
- scikit-learn
- matplotlib
- tensorflow (with keras)
- umap-learn

## Usage

1. Place the dataset `synthetic_data.csv` in the `data/` directory.
2. Run the script to perform the grid search and find the best autoencoder architecture and clustering parameters.
3. The best parameters and their corresponding silhouette score will be printed.
4. Visualize the clustering results using PCA, t-SNE, and UMAP.

```bash
python autoencoder_clustering.py
