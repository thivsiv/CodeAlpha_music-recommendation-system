# Music Recommendation System

A machine learning system that predicts the likelihood of users repeatedly listening to songs within a month, inspired by Spotify's recommendation engine. Generates personalized song recommendations using temporal patterns and user listening history.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-green)](https://scikit-learn.org)

## Features

- 🎵 **Synthetic Data Generation**: Realistic user listening patterns with temporal distribution
- ⏳ **Temporal Feature Engineering**: Hour/day/week features + 30-day listening windows
- 🧠 **ML Model**: Random Forest classifier with class balancing
- 🎯 **Personalized Recommendations**: User-specific suggestions with genre diversity
- 📊 **Evaluation Metrics**: Accuracy, PR-AUC, and feature importance visualization

## Installation

1. **Clone Repository**
```bash
git clone https://github.com/thivsiv/codealpha_tasks-music-recommendation-system.git
cd music-recommendation-system
```

2. **Create Virtual Environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
3.**Install Dependencies**
```bash
pip install -r requirements.txt
```

Create requirements.txt with:
```bash
pandas>=1.5.3
scikit-learn>=1.2.2
numpy>=1.24.3
matplotlib>=3.7.1
python-dateutil>=2.8.2
```

## Usage

1.**Run Recommendation System**
```bash
python recommendation_system.py
```
2.**Expected Output**
```bash
Generating synthetic data...
Training model...

Model Evaluation:
Accuracy: 0.82
PR AUC: 0.75

Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      1523
           1       0.72      0.61      0.66       577

    accuracy                           0.82      2100
   macro avg       0.79      0.76      0.77      2100
weighted avg       0.81      0.82      0.81      2100

Generating recommendations...

Top 10 Recommendations:
 1. Song 342   (Rock)       Repeat Probability: 89.23%
 2. Song 215   (Electronic) Repeat Probability: 85.41%
 3. Song 478   (Hip-Hop)    Repeat Probability: 83.17%
 ...
```
## Code Structure
```bash
music-recommendation-system/
├── data/                   # Generated dataset
├── venv/                   # Virtual environment
├── recommendation_system.py  # Main application code
├── requirements.txt        # Dependency list
└── README.md               # This documentation
```
### Key Components

1. Synthetic Data Generation
- User listening patterns
- Temporal distribution of plays
- 30-day repeat listening simulation
   
2. Feature Engineering
- Time decay: exp(-days/30)
- User engagement scores
- Song trend calculations

3. Recommendation Engine
- Filters previously heard songs
- Genre diversity enforcement
- Time-contextual predictions

## Customization

1. Adjust Synthetic Data
```bash
# Generate larger dataset
df = generate_synthetic_data(num_users=5000, num_songs=2000, num_records=100000)
```
2. Modify Model Parameters
  ```bash
self.model = RandomForestClassifier(
    n_estimators=300,  # Increase tree count
    max_depth=15,      # Deeper trees
    class_weight={0:1, 1:3}  # Custom class weights
)
``` 

## Acknowledgments

- Inspired by Spotify's recommendation system
- Synthetic data design from RecSys challenge patterns
- Scikit-learn machine learning framework
