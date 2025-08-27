"""
F1 Race Prediction Model
A comprehensive machine learning model for Formula 1 race predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# 2025 F1 Driver-Team Lineup
DRIVER_TEAM_2025 = {
    'Lewis Hamilton': 'Ferrari',
    'Charles Leclerc': 'Ferrari',
    'Max Verstappen': 'Red Bull Racing Honda RBPT',
    'Sergio Pérez': 'Red Bull Racing Honda RBPT',
    'Lando Norris': 'McLaren Mercedes',
    'Oscar Piastri': 'McLaren Mercedes',
    'George Russell': 'Mercedes',
    'Kimi Antonelli': 'Mercedes',
    'Fernando Alonso': 'Aston Martin Aramco Mercedes',
    'Lance Stroll': 'Aston Martin Aramco Mercedes',
    'Pierre Gasly': 'Alpine Renault',
    'Jack Doohan': 'Alpine Renault',
    'Alex Albon': 'Williams Mercedes',
    'Franco Colapinto': 'Williams Mercedes',
    'Yuki Tsunoda': 'RB Honda RBPT',
    'Isack Hadjar': 'RB Honda RBPT',
    'Nico Hülkenberg': 'Kick Sauber Ferrari',
    'Gabriel Bortoleto': 'Kick Sauber Ferrari',
    'Kevin Magnussen': 'Haas Ferrari',
    'Esteban Ocon': 'Haas Ferrari'
}


class F1Predictor:
    """
    F1 Race Prediction Model with advanced feature engineering
    """
    
    def __init__(self):
        self.models = {}
        self.encoders = {
            'driver': LabelEncoder(),
            'team': LabelEncoder()
        }
        self.feature_columns = []
        self.is_trained = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features for improved prediction accuracy
        
        Args:
            df (pd.DataFrame): Raw historical data
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        print("🛠️ Engineering advanced features...")
        
        # Sort chronologically for proper rolling calculations
        df = df.sort_values(by=['year', 'race'], kind='mergesort').reset_index(drop=True)
        
        # Encode categorical variables
        df['driver_encoded'] = self.encoders['driver'].fit_transform(df['driver'])
        df['team_encoded'] = self.encoders['team'].fit_transform(df['team'])
        
        # Basic performance metrics (using expanding window to avoid lookahead bias)
        df['driver_avg_finish'] = df.groupby('driver')['finish_pos'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(10.5)
        )
        df['driver_avg_quali'] = df.groupby('driver')['quali_pos'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(10.5)
        )
        df['team_avg_finish'] = df.groupby('team')['finish_pos'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(10.5)
        )
        
        # Advanced performance metrics
        df['driver_form_ewma'] = df.groupby('driver')['finish_pos'].transform(
            lambda x: x.ewm(span=5, adjust=False).mean().shift(1).fillna(10.5)
        )
        
        # Race craft (ability to gain/lose positions during race)
        df['race_craft'] = df['quali_pos'] - df['finish_pos']
        df['avg_race_craft'] = df.groupby('driver')['race_craft'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0)
        )
        
        # Recent points momentum
        df['points_momentum'] = df.groupby('driver')['points'].transform(
            lambda x: x.rolling(5, min_periods=1).sum().shift(1).fillna(0)
        )
        
        # Circuit-specific performance
        df['driver_circuit_avg_finish'] = df.groupby(['driver', 'race'])['finish_pos'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['team_circuit_avg_finish'] = df.groupby(['team', 'race'])['finish_pos'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Fill NaN values for circuit-specific features
        df['driver_circuit_avg_finish'].fillna(df['driver_avg_finish'], inplace=True)
        df['team_circuit_avg_finish'].fillna(df['team_avg_finish'], inplace=True)
        
        # Teammate battle statistics
        df['race_id'] = df['year'].astype(str) + '_' + df['race']
        df['teammate_finish_pos'] = df.groupby(['race_id', 'team'])['finish_pos'].transform(
            lambda x: x.shift(-1).fillna(x.shift(1))
        )
        df['beat_teammate'] = (df['finish_pos'] < df['teammate_finish_pos']).astype(int)
        df['teammate_win_rate'] = df.groupby('driver')['beat_teammate'].transform(
            lambda x: x.rolling(10, min_periods=1).mean().shift(1).fillna(0.5)
        )
        
        # Reliability metrics
        df['is_dnf'] = (df['finish_pos'] >= 21).astype(int)
        df['driver_dnf_rate'] = df.groupby('driver')['is_dnf'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0)
        )
        df['team_dnf_rate'] = df.groupby('team')['is_dnf'].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0)
        )
        
        # Clean up temporary columns
        df.drop(columns=['race_craft', 'race_id', 'teammate_finish_pos', 'beat_teammate', 'is_dnf'], 
                inplace=True)
        
        # Define feature columns for model training
        self.feature_columns = [
            'driver_encoded', 'team_encoded', 'quali_pos',
            'driver_avg_finish', 'driver_avg_quali', 'team_avg_finish',
            'driver_form_ewma', 'avg_race_craft', 'points_momentum',
            'driver_circuit_avg_finish', 'team_circuit_avg_finish',
            'teammate_win_rate', 'driver_dnf_rate', 'team_dnf_rate'
        ]
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train prediction models using the engineered features
        
        Args:
            df (pd.DataFrame): Dataframe with engineered features
            
        Returns:
            Dict[str, float]: Model performance metrics
        """
        print("🤖 Training prediction models...")
        
        # Prepare features
        X = df[self.feature_columns].fillna(10.5)
        
        metrics = {}
        
        # Train win predictor
        y_win = df['won']
        if y_win.sum() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y_win, test_size=0.2, random_state=42)
            
            self.models['win'] = RandomForestClassifier(
                n_estimators=200, 
                random_state=42, 
                max_depth=12, 
                min_samples_leaf=3
            )
            self.models['win'].fit(X_train, y_train)
            
            train_acc = self.models['win'].score(X_train, y_train)
            test_acc = self.models['win'].score(X_test, y_test)
            metrics['win_train_acc'] = train_acc
            metrics['win_test_acc'] = test_acc
            
            print(f"  🏆 Win predictor - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
        
        # Train podium predictor
        y_podium = df['podium']
        X_train, X_test, y_train, y_test = train_test_split(X, y_podium, test_size=0.2, random_state=42)
        
        self.models['podium'] = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            max_depth=12, 
            min_samples_leaf=3
        )
        self.models['podium'].fit(X_train, y_train)
        
        train_acc = self.models['podium'].score(X_train, y_train)
        test_acc = self.models['podium'].score(X_test, y_test)
        metrics['podium_train_acc'] = train_acc
        metrics['podium_test_acc'] = test_acc
        
        print(f"  🥇 Podium predictor - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
        
        # Train finish position predictor
        y_finish = df['finish_pos']
        X_train, X_test, y_train, y_test = train_test_split(X, y_finish, test_size=0.2, random_state=42)
        
        self.models['finish'] = RandomForestRegressor(
            n_estimators=200, 
            random_state=42, 
            max_depth=12, 
            min_samples_leaf=3
        )
        self.models['finish'].fit(X_train, y_train)
        
        train_rmse = np.sqrt(((self.models['finish'].predict(X_train) - y_train)**2).mean())
        test_rmse = np.sqrt(((self.models['finish'].predict(X_test) - y_test)**2).mean())
        metrics['finish_train_rmse'] = train_rmse
        metrics['finish_test_rmse'] = test_rmse
        
        print(f"  🏁 Finish predictor - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
        
        self.is_trained = True
        print("✅ Models trained successfully!")
        
        return metrics
    
    def predict_race(self, 
                     race_name: str,
                     driver_team_dict: Optional[Dict[str, str]] = None,
                     historical_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict race results for given drivers and teams
        
        Args:
            race_name (str): Name of the race
            driver_team_dict (Dict[str, str]): Driver to team mapping
            historical_data (pd.DataFrame): Historical data for context
            
        Returns:
            pd.DataFrame: Predictions for all drivers across different starting positions
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if driver_team_dict is None:
            driver_team_dict = DRIVER_TEAM_2025
        
        print(f"🔮 Generating predictions for {race_name}...")
        
        # Prepare historical stats
        if historical_data is not None:
            # Check if historical_data has engineered features, if not, engineer them
            if 'avg_race_craft' not in historical_data.columns:
                historical_data = self.engineer_features(historical_data.copy())
            
            driver_latest_stats = historical_data.loc[
                historical_data.groupby('driver').idxmax()['year']
            ].set_index('driver')
            
            team_latest_stats = historical_data.loc[
                historical_data.groupby('team').idxmax()['year']
            ].set_index('team')
            
            # Grid averages for new drivers
            grid_avg_finish = historical_data['finish_pos'].mean()
            grid_avg_quali = historical_data['quali_pos'].mean()
            grid_avg_race_craft = historical_data['avg_race_craft'].mean() if 'avg_race_craft' in historical_data.columns else 0
            grid_avg_dnf = historical_data['driver_dnf_rate'].mean() if 'driver_dnf_rate' in historical_data.columns else 0.15
            
            # Race-specific history
            race_history = historical_data[
                historical_data['race'].str.contains(race_name.split()[0], case=False, na=False)
            ]
        else:
            # Default values if no historical data provided
            driver_latest_stats = pd.DataFrame()
            team_latest_stats = pd.DataFrame()
            grid_avg_finish = 10.5
            grid_avg_quali = 10.5
            grid_avg_race_craft = 0
            grid_avg_dnf = 0.15
            race_history = pd.DataFrame()
        
        predictions = []
        
        # Team mapping for historical consistency
        team_mapping = {
            'Ferrari': 'Scuderia Ferrari',
            'Red Bull Racing Honda RBPT': 'Red Bull Racing Honda RBPT',
            'McLaren Mercedes': 'McLaren Mercedes',
            'Mercedes': 'Mercedes',
            'Aston Martin Aramco Mercedes': 'Aston Martin Aramco Mercedes',
            'Alpine Renault': 'Alpine Renault',
            'Williams Mercedes': 'Williams Mercedes',
            'RB Honda RBPT': 'AlphaTauri Honda RBPT',
            'Kick Sauber Ferrari': 'Alfa Romeo Ferrari',
            'Haas Ferrari': 'Haas Ferrari'
        }
        
        for driver_2025, team_2025 in driver_team_dict.items():
            # Get driver stats or use defaults for new drivers
            is_rookie = driver_2025 not in driver_latest_stats.index
            
            if not is_rookie:
                stats = driver_latest_stats.loc[driver_2025]
                driver_avg_finish = stats.get('driver_avg_finish', grid_avg_finish)
                driver_avg_quali = stats.get('driver_avg_quali', grid_avg_quali)
                driver_form_ewma = stats.get('driver_form_ewma', grid_avg_finish)
                avg_race_craft = stats.get('avg_race_craft', grid_avg_race_craft)
                points_momentum = stats.get('points_momentum', 0)
                teammate_win_rate = stats.get('teammate_win_rate', 0.5)
                driver_dnf_rate = stats.get('driver_dnf_rate', grid_avg_dnf)
            else:
                driver_avg_finish = grid_avg_finish
                driver_avg_quali = grid_avg_quali
                driver_form_ewma = grid_avg_finish
                avg_race_craft = grid_avg_race_craft
                points_momentum = 0
                teammate_win_rate = 0.5
                driver_dnf_rate = grid_avg_dnf
            
            # Get team stats
            historical_team = team_mapping.get(team_2025, team_2025)
            if historical_team in team_latest_stats.index:
                team_stats = team_latest_stats.loc[historical_team]
                team_avg_finish = team_stats.get('team_avg_finish', grid_avg_finish)
                team_dnf_rate = team_stats.get('team_dnf_rate', grid_avg_dnf)
            else:
                team_avg_finish = grid_avg_finish
                team_dnf_rate = grid_avg_dnf
            
            # Circuit-specific performance
            if not race_history.empty:
                driver_circuit_history = race_history[race_history['driver'] == driver_2025]
                driver_circuit_avg_finish = (driver_circuit_history['finish_pos'].mean() 
                                           if not driver_circuit_history.empty else driver_avg_finish)
                
                team_circuit_history = race_history[race_history['team'] == historical_team]
                team_circuit_avg_finish = (team_circuit_history['finish_pos'].mean() 
                                         if not team_circuit_history.empty else team_avg_finish)
            else:
                driver_circuit_avg_finish = driver_avg_finish
                team_circuit_avg_finish = team_avg_finish
            
            # Encode driver and team
            try:
                driver_encoded = self.encoders['driver'].transform([driver_2025])[0]
            except ValueError:
                driver_encoded = -1  # New driver
            
            try:
                team_encoded = self.encoders['team'].transform([historical_team])[0]
            except ValueError:
                team_encoded = -1  # New team
            
            # Generate predictions for different qualifying positions
            for quali_pos in range(1, 21):
                pred_features = np.array([[
                    driver_encoded, team_encoded, quali_pos,
                    driver_avg_finish, driver_avg_quali, team_avg_finish,
                    driver_form_ewma, avg_race_craft, points_momentum,
                    driver_circuit_avg_finish, team_circuit_avg_finish,
                    teammate_win_rate, driver_dnf_rate, team_dnf_rate
                ]])
                
                # Get predictions
                win_prob = (self.models['win'].predict_proba(pred_features)[0][1] * 100 
                           if 'win' in self.models else 5.0)
                podium_prob = self.models['podium'].predict_proba(pred_features)[0][1] * 100
                finish_pred = self.models['finish'].predict(pred_features)[0]
                
                # Apply heuristic adjustments for major changes
                if driver_2025 == 'Lewis Hamilton' and team_2025 == 'Ferrari':
                    win_prob *= 1.2
                    podium_prob = min(99, podium_prob * 1.15)
                
                predictions.append({
                    'driver': driver_2025,
                    'team': team_2025,
                    'quali_pos': quali_pos,
                    'win_prob': win_prob,
                    'podium_prob': podium_prob,
                    'predicted_finish': finish_pred,
                    'is_rookie': is_rookie
                })
        
        return pd.DataFrame(predictions)
    
    def get_race_winners(self, predictions_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get top race winner predictions from optimal starting positions
        
        Args:
            predictions_df (pd.DataFrame): Full predictions dataframe
            top_n (int): Number of top predictions to return
            
        Returns:
            pd.DataFrame: Top winner predictions
        """
        # Get best predictions for each driver
        top_winners = predictions_df.loc[predictions_df.groupby('driver')['win_prob'].idxmax()]
        return top_winners.sort_values('win_prob', ascending=False).head(top_n)
    
    def get_podium_predictions(self, predictions_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get top podium predictions from optimal starting positions
        
        Args:
            predictions_df (pd.DataFrame): Full predictions dataframe
            top_n (int): Number of top predictions to return
            
        Returns:
            pd.DataFrame: Top podium predictions
        """
        # Get best predictions for each driver
        top_podium = predictions_df.loc[predictions_df.groupby('driver')['podium_prob'].idxmax()]
        return top_podium.sort_values('podium_prob', ascending=False).head(top_n)


def create_f1_predictor(historical_data: pd.DataFrame) -> F1Predictor:
    """
    Create and train an F1 predictor with historical data
    
    Args:
        historical_data (pd.DataFrame): Historical F1 race data
        
    Returns:
        F1Predictor: Trained prediction model
    """
    predictor = F1Predictor()
    
    # Engineer features
    df_with_features = predictor.engineer_features(historical_data.copy())
    
    # Train models
    metrics = predictor.train_models(df_with_features)
    
    return predictor, metrics


if __name__ == "__main__":
    print("🏎️ F1 Predictor - Standalone Test")
    print("This module requires historical data to function properly.")
    print("Use examples/run_predictions.py for full functionality.")
