import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, precision_recall_curve, auc)
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

def generate_synthetic_data(num_users=1000, num_songs=500, num_records=50000):
    np.random.seed(42)
    
    # Time range setup
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=6)
    
    # Correct timestamp generation
    timestamps = [
        start_date + timedelta(
            days=np.random.randint(0, 180),
            hours=np.random.randint(0, 24)
        ) 
        for _ in range(num_records)
    ]
    
    # Base dataframe
    data = {
        'user_id': np.random.randint(1, num_users+1, num_records),
        'song_id': np.random.randint(1, num_songs+1, num_records),
        'timestamp': sorted(timestamps, reverse=True),
        'song_duration': np.random.randint(120, 600, num_records),
        'genre': np.random.choice(['Pop', 'Rock', 'Electronic', 'Hip-Hop', 'Jazz'], num_records),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable
    df['listen_count'] = np.random.poisson(3, num_records) + 1
    df['last_30_days'] = df['timestamp'] > (end_date - timedelta(days=30))

    repeat_mask = (
    (df['last_30_days']) & 
    (df['listen_count'] > np.random.randint(2, 4, num_records))
)

  
    
    df['repeated_listen'] = np.where(repeat_mask, 1, 0)
    
    # Ensure minimum 25% positive class
    positive_rate = df['repeated_listen'].mean()
    if positive_rate < 0.25:
        needed = int(0.25 * len(df)) - sum(df['repeated_listen'])
        positive_samples = df[df['last_30_days']].sample(needed, replace=True)
        positive_samples['repeated_listen'] = 1
        df = pd.concat([df, positive_samples]).sample(frac=1).reset_index(drop=True)
    
    return df.drop('last_30_days', axis=1)

def create_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    end_date = datetime.now()
    
    # Temporal features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['days_since_play'] = (end_date - df['timestamp']).dt.days
    
    # 30-day window features
    df['play_in_last_30d'] = (df['days_since_play'] <= 30).astype(int)
    df['time_decay'] = np.exp(-df['days_since_play']/30)
    
    # User behavior features
    user_features = df.groupby('user_id').agg({
        'listen_count': ['mean', 'sum'],
        'days_since_play': ['min', 'max'],
        'play_in_last_30d': 'sum',
        'repeated_listen': 'mean'
    }).reset_index()
    user_features.columns = [
        'user_id', 'user_avg_listen', 'user_total_listen',
        'user_last_play', 'user_first_play', 'user_30d_plays',
        'user_repeat_rate'
    ]
    
    # Song features
    song_features = df.groupby('song_id').agg({
        'listen_count': ['mean', 'sum'],
        'days_since_play': 'mean',
        'play_in_last_30d': 'sum',
        'repeated_listen': 'mean'
    }).reset_index()
    song_features.columns = [
        'song_id', 'song_avg_listen', 'song_total_listen',
        'song_recency', 'song_30d_plays', 'song_repeat_rate'
    ]
    
    # Merge features
    df = df.merge(user_features, on='user_id', how='left')
    df = df.merge(song_features, on='song_id', how='left')
    
    # Fill missing values
    df.fillna({
        'user_avg_listen': 0,
        'user_total_listen': 0,
        'user_30d_plays': 0,
        'user_repeat_rate': 0,
        'song_avg_listen': 0,
        'song_total_listen': 0,
        'song_30d_plays': 0,
        'song_repeat_rate': 0
    }, inplace=True)
    
    # Derived features
    df['user_engagement'] = df['user_30d_plays'] * df['time_decay']
    df['song_trend'] = df['song_30d_plays'] / (df['song_total_listen'] + 1)
    
    return df

class MusicRecommender:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            max_depth=10
        )
        self.feature_columns = [
            'listen_count', 'song_duration', 'hour_of_day',
            'day_of_week', 'days_since_play', 'time_decay',
            'user_avg_listen', 'user_total_listen', 'user_30d_plays',
            'user_repeat_rate', 'song_avg_listen', 'song_total_listen',
            'song_recency', 'song_30d_plays', 'song_repeat_rate',
            'user_engagement', 'song_trend'
        ]
        self.genres = []
        self.genre_columns = []
        self.training_columns = []
    
    def train(self, df):
        # Time-based validation
        tscv = TimeSeriesSplit(n_splits=3)
        df = df.sort_values('timestamp')
        
        # Store genres
        self.genres = df['genre'].unique().tolist()
        
        # Create dummy variables
        df = pd.get_dummies(df, columns=['genre'], prefix='genre')
        self.genre_columns = [col for col in df.columns if col.startswith('genre_')]
        self.training_columns = self.feature_columns + self.genre_columns
        
        # Prepare data
        X = df[self.training_columns]
        y = df['repeated_listen']
        
        # Temporal split
        for train_index, test_index in tscv.split(X):
            self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
            self.y_train, self.y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate(self):
        if len(np.unique(self.y_test)) < 2:
            print("Insufficient class variety for evaluation")
            return
            
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)
        
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
            print(f"PR AUC: {auc(recall, precision):.2f}")
        else:
            print("PR AUC: Not available (single class prediction)")
            
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Feature importance
        plt.figure(figsize=(12, 8))
        feat_importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns)
        feat_importances.nlargest(15).plot(kind='barh')
        plt.title('Top 15 Feature Importances')
        plt.show()
    
    def recommend(self, user_history, all_songs, current_time=datetime.now()):
        # Filter unheard songs
        heard_songs = user_history['song_id'].unique()
        candidate_songs = all_songs[~all_songs['song_id'].isin(heard_songs)]
        
        # User context
        user_features = user_history.iloc[0][[
            'user_avg_listen', 'user_total_listen', 'user_30d_plays',
            'user_repeat_rate', 'user_engagement'
        ]].to_dict()
        
        predictions = []
        for _, song in candidate_songs.iterrows():
            # Temporal features
            last_play_days = (current_time - user_history['timestamp'].max()).days
            time_decay = np.exp(-last_play_days/30)
            
            # Build features
            features = {
                'listen_count': user_history['listen_count'].mean(),
                'song_duration': song['song_duration'],
                'hour_of_day': current_time.hour,
                'day_of_week': current_time.weekday(),
                'days_since_play': last_play_days,
                'time_decay': time_decay,
                **user_features,
                'song_avg_listen': song['song_avg_listen'],
                'song_total_listen': song['song_total_listen'],
                'song_recency': song['song_recency'],
                'song_30d_plays': song['song_30d_plays'],
                'song_repeat_rate': song['song_repeat_rate'],
                'song_trend': song['song_trend'],
            }
            
            # Add genre features
            song_genre = song['genre']
            for genre in self.genres:
                features[f'genre_{genre}'] = 1 if genre == song_genre else 0
            
            # Create feature dataframe
            features_df = pd.DataFrame([features])
            
            # Ensure all columns exist
            missing_cols = set(self.training_columns) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0
                
            features_df = features_df[self.training_columns]
            
            # Predict probability
            prob = self.model.predict_proba(features_df)[0][1]
            predictions.append((song['song_id'], prob, song['genre']))
        
        return self._ensure_diversity(predictions)
    
    def _ensure_diversity(self, recommendations, max_per_genre=2):
        final_rec = []
        genre_counts = {}
        
        # Sort by probability
        sorted_recs = sorted(recommendations, key=lambda x: -x[1])
        
        for rec in sorted_recs:
            genre = rec[2]
            current_count = genre_counts.get(genre, 0)
            
            if current_count < max_per_genre:
                final_rec.append(rec)
                genre_counts[genre] = current_count + 1
                
            if len(final_rec) >= 10:
                break
                
        return final_rec

if __name__ == "__main__":
    # Data pipeline
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_records=10000)
    df = create_features(df)
    
    # Model training
    print("\nTraining model...")
    recommender = MusicRecommender()
    recommender.train(df)
    
    # Evaluation
    print("\nModel Evaluation:")
    recommender.evaluate()
    
    # Generate recommendations
    sample_user = df[df['user_id'] == 42].sort_values('timestamp', ascending=False).iloc[0:1]
    all_songs = df[['song_id', 'song_duration', 'genre',
                    'song_avg_listen', 'song_total_listen', 'song_repeat_rate',
                    'song_recency', 'song_30d_plays', 'song_trend']].drop_duplicates()
    
    print("\nGenerating recommendations...")
    recommendations = recommender.recommend(sample_user, all_songs)
    
    print("\nTop 10 Recommendations:")
    for idx, (song_id, prob, genre) in enumerate(recommendations, 1):
        print(f"{idx:>2}. Song {song_id:<5} ({genre:<10}) Repeat Probability: {prob:.2%}")

# Install dependencies: pip install pandas scikit-learn numpy matplotlib python-dateutil