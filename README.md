# Team Activity Clustering and Performance Analysis

This project aims to analyze team activities, cluster teams based on their strengths and weaknesses, and evaluate the performance of these clusters using various machine learning techniques. The analysis includes preprocessing data, performing clustering, optimizing model parameters, training a RandomForest classifier, generating artificial data for evaluation, and visualizing the results.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Functions and Methodology](#functions-and-methodology)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Overview

This project utilizes clustering algorithms to group teams based on their activity metrics and strengths in various fields. The clustering results are then evaluated using a supervised learning approach with a RandomForest classifier, which is optimized using GridSearchCV. The project also includes generating an artificial dataset for validation purposes and visualizing the results.

## Data Description

The data consists of team activity metrics, which are categorized into involvement and industry columns. Each team's activities are qualitatively described as 'Strong', 'Good', 'Average', or 'None'. The dataset includes columns for various activities such as Funding, Application-Oriented, Demos, Industrial Collaborations, System Maturity, Number of Members, Academic Collaborations, and several industry-specific columns.

## Setup Instructions

1. Clone the repository to your local machine.
2. Ensure you have Python 3.8 or later installed.
3. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
