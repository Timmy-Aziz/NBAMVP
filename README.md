# NBA MVP Predictor

## Project Overview
This project aims to develop a predictive model using machine learning to identify potential NBA MVP candidates based on historical player performance data. The model is built using a Random Forest classifier and is trained to predict the likelihood of a player winning the MVP award in a given season.

## Dataset
The dataset used for this project includes a variety of player statistics and performance metrics across multiple NBA seasons. Key features include points per game, assists, rebounds, player efficiency rating (PER), win shares (WS), and value over replacement player (VORP).

## Features and Target Variable
- **Features (X)**: The dataset contains several numerical features that represent player performance and impact on the game, such as `ws_per_48`, `vorp`, `per`, etc.
- **Target Variable (Y)**: The target variable is binary and represents whether a player was the MVP (`1`) or not (`0`) in a given season.

## Methodology
The project involves several key steps:
1. Data Preprocessing: Cleaning the data and handling categorical variables with encoding techniques.
2. Feature Engineering: Creating new features that could have predictive power for MVP candidacy.
3. Model Training: Using a Random Forest classifier to build the predictive model.
4. Hyperparameter Tuning: Applying Grid Search to find the best model parameters.
5. SMOTE: Utilizing Synthetic Minority Over-sampling Technique to address class imbalance in the dataset. Note that two models are made with and without SMOTE for different aggression on model prediction
6. Evaluation: Assessing the model's performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## Results
The final models demonstrate the ability to accurately identify MVP candidates with an emphasis on maximizing recall to ensure potential MVPs are not missed. It showcases which statistical categories are most important in determining the MVP

## Usage
To run the model prediction, use the following command:

```bash
python mvp_predictor.py
