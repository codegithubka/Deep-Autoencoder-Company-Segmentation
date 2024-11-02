# Team Strength Clustering with Autoencoders

This repository contains a Python script to cluster teams based on their strengths using autoencoders and various clustering techniques. The script includes preprocessing steps, feature engineering, hyperparameter tuning, model training, and cluster evaluation.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Functions](#functions)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Clustering Evaluation](#clustering-evaluation)
7. [Visualization](#visualization)
8. [Future Improvements](#future-improvements)

## Introduction

This project aims to group teams based on their strengths across different industries. The script leverages autoencoders to reduce dimensionality and applies clustering algorithms to discover meaningful groupings. The performance of the clusters is evaluated using metrics such as silhouette score and Davies-Bouldin index.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset and place it in the appropriate location. Ensure the dataset has the required structure as shown in `synthetic_data.csv`.

2. Run the script:
    ```bash
    python main_script.py
    ```

## Hyperparameter Tuning

The script uses a grid search over a range of hyperparameters to find the best configuration for the autoencoder. The hyperparameters include:
- Layer sizes
- Encoding dimension
- Dropout rate
- Batch size
- Number of epochs

## Clustering Evaluation

The script evaluates the clustering performance using the following metrics:
- **Silhouette Score**: Measures how similar a point is to its own cluster compared to other clusters.
- **Davies-Bouldin Index**: Measures the average similarity ratio of each cluster with the one most similar to it.
- **Elbow Method**: Uses the KneeLocator to find the optimal number of clusters based on inertia.

## Visualization

The script includes functions to visualize the training history, cluster statistics, and cluster distribution. The visualizations help in understanding the model's performance and the characteristics of each cluster.

