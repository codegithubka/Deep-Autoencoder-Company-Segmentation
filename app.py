from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)


# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, index_col=0).transpose()
    weights = {'Strong': 3, 'Good': 2, 'Average': 1, 'None': 0}

    involvement_columns = [
        'Security', 'Humanities', 'Nat. Sci', 'Health', 'AI Ethics', 'Big Data',
        'Robotics', 'Documents', 'Multimedia', 'NLP', 'KRR', 'Graphs', 'DL/ML',
        'Funding', 'Application-Oriented', 'Number of Members',
        'Academic Collaborations', 'System Maturity', 'Demos', 'Industrial Collaborations'
    ]

    for column in involvement_columns:
        if column in df.columns:
            df[column] = df[column].map(weights).fillna(0)

    return df


# Calculate pairwise similarity
def calculate_pairwise_similarity(data):
    industry_cols = ['Security', 'Humanities', 'Nat. Sci', 'Health', 'AI Ethics', 'Big Data',
                     'Robotics', 'Documents', 'Multimedia', 'NLP', 'KRR', 'Graphs', 'DL/ML']

    sim_matrix = np.zeros((len(data), len(data)))

    scoring_matrix = np.array([
        [0, 1, 2, 3],  # None
        [1, 2, 3, 4],  # Average
        [2, 3, 4, 5],  # Good
        [3, 4, 5, 6]  # Strong
    ])

    involvement_index = {'None': 0, 'Average': 1, 'Good': 2, 'Strong': 3}
    reverse_weights = {0: 'None', 1: 'Average', 2: 'Good', 3: 'Strong'}

    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                similarity_score = 0
                for col in industry_cols:
                    level_i = reverse_weights[data[col].iloc[i]]
                    level_j = reverse_weights[data[col].iloc[j]]
                    index_i = involvement_index[level_i]
                    index_j = involvement_index[level_j]
                    similarity_score += scoring_matrix[index_i][index_j]
                sim_matrix[i][j] = similarity_score

    return sim_matrix


# Calculate complementary scores
def calculate_complementary_scores(data):
    comp_cols = ['Number of Members', 'Application-Oriented', 'Academic Collaborations',
                 'System Maturity', 'Demos', 'Industrial Collaborations']

    data = data[comp_cols]
    comp_matrix = np.zeros((len(data), len(data)))

    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                complementary_score = 0
                for col in comp_cols:
                    complementary_score += abs(data[col].iloc[i] - data[col].iloc[j])
                comp_matrix[i][j] = complementary_score

    max_score = np.max(comp_matrix)
    comp_matrix = max_score - comp_matrix

    return comp_matrix


# Combine similarity and complementary scores
def combine_scores(industry_similarity, compl_scores, alpha=0.5):
    if not (0 <= alpha <= 1):
        raise ValueError("alpha should be between 0 and 1")

    combined_matrix = alpha * industry_similarity + (1 - alpha) * compl_scores
    return combined_matrix


# Apply clustering
def apply_clustering(sim_matrix, n_clusters):
    distance_matrix = 1 - (sim_matrix / np.max(sim_matrix))
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
    labels = clustering.fit_predict(distance_matrix)
    return labels


# Calculate cluster averages
def calculate_cluster_averages(df, labels):
    df['cluster'] = labels
    cluster_averages = df.groupby('cluster').mean().drop(columns='cluster', errors='ignore')
    return cluster_averages


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_plot_data', methods=['POST'])
def get_plot_data():
    file_path = request.form['file_path']
    df = load_and_preprocess_data(file_path)

    # Calculate similarity and complementary scores
    industry_scores = calculate_pairwise_similarity(df)
    complementary_scores = calculate_complementary_scores(df)
    combined_scores = combine_scores(industry_scores, complementary_scores)

    # Apply clustering
    num_clusters = 5
    clustering_labels = apply_clustering(combined_scores, num_clusters)

    # Calculate cluster averages
    cluster_averages = calculate_cluster_averages(df, clustering_labels)

    # Create a heatmap of combined scores
    fig_combined = px.imshow(combined_scores, labels=dict(x="Team 1", y="Team 2", color="Combined Score"),
                             x=list(df.index), y=list(df.index), title="Combined Scores Heatmap")
    graphJSON_combined = pio.to_json(fig_combined)

    # Combine clustering results into a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Team': df.index,
        'Cluster Label': clustering_labels
    })

    # Return both plots and comparison data
    return jsonify({
        'combined_heatmap': graphJSON_combined,
        'comparison_data': comparison_df.to_dict(orient='records')
    })


if __name__ == '__main__':
    app.run(debug=True,port=5001)
