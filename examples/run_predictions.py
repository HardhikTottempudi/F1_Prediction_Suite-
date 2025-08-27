#!/usr/bin/env python3
"""
F1 Prediction Suite - Example Usage
Demonstrates how to use the F1 prediction system for race forecasting
"""

import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from data_loader import F1DataLoader
    from f1_predictor import F1Predictor, DRIVER_TEAM_2025, create_f1_predictor
    import pandas as pd
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root with activated environment")
    sys.exit(1)


def display_predictions(predictions_df, predictor, race_name):
    """Display formatted prediction results"""
    
    print(f"\n🏆 {race_name.upper()} RACE WINNER PREDICTIONS")
    print("=" * 80)
    
    # Get top winners
    top_winners = predictor.get_race_winners(predictions_df, top_n=10)
    
    print(f"{'Driver':<18} {'Team':<25} {'Start':<6} {'Win%':<6} {'Podium%':<8}")
    print("-" * 80)
    
    for _, pred in top_winners.iterrows():
        rookie_indicator = " 🆕" if pred.get('is_rookie', False) else ""
        print(f"{pred['driver']:<18} {pred['team']:<25} P{pred['quali_pos']:<5} "
              f"{pred['win_prob']:<6.1f} {pred['podium_prob']:<8.1f}{rookie_indicator}")
    
    print(f"\n🥇 TOP PODIUM CONTENDERS")
    print("-" * 80)
    
    # Get top podium finishers
    top_podium = predictor.get_podium_predictions(predictions_df, top_n=10)
    
    print(f"{'Driver':<18} {'Team':<25} {'Start':<6} {'Podium%':<8} {'Win%':<6}")
    print("-" * 80)
    
    for _, pred in top_podium.iterrows():
        rookie_indicator = " 🆕" if pred.get('is_rookie', False) else ""
        print(f"{pred['driver']:<18} {pred['team']:<25} P{pred['quali_pos']:<5} "
              f"{pred['podium_prob']:<8.1f} {pred['win_prob']:<6.1f}{rookie_indicator}")
    
    # Final prediction summary
    top_prediction = top_winners.iloc[0]
    print("\n" + "=" * 80)
    print(f"🏁 {race_name.upper()} - FINAL PREDICTION")
    print("=" * 80)
    print(f"🏆 RACE WINNER PREDICTION: {top_prediction['driver']}")
    print(f"   🏎️ Team: {top_prediction['team']}")
    print(f"   📍 Optimal Starting Position: P{top_prediction['quali_pos']}")
    print(f"   🎯 Win Probability: {top_prediction['win_prob']:.1f}%")
    print(f"   🥇 Podium Probability: {top_prediction['podium_prob']:.1f}%")
    
    print(f"\n🏁 PREDICTED {race_name.upper()} PODIUM:")
    for i, (_, pred) in enumerate(top_winners.head(3).iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"   {medal} {i}{'st' if i==1 else 'nd' if i==2 else 'rd'}: "
              f"{pred['driver']} ({pred['team']})")
    
    # Special analysis for Lewis Hamilton at Ferrari
    hamilton_pred = predictions_df[predictions_df['driver'] == 'Lewis Hamilton'].nlargest(1, 'win_prob')
    if not hamilton_pred.empty:
        hamilton_best = hamilton_pred.iloc[0]
        print(f"\n🔴 LEWIS HAMILTON @ FERRARI ANALYSIS:")
        print(f"   Best chance from P{hamilton_best['quali_pos']}: "
              f"{hamilton_best['win_prob']:.1f}% win, {hamilton_best['podium_prob']:.1f}% podium")


def main():
    """Main prediction function"""
    print("🏎️ F1 PREDICTION SUITE")
    print("=" * 50)
    print("Formula 1 Race Prediction System")
    print("Powered by machine learning and 5+ years of historical data")
    print("=" * 50)
    
    # Check for cached data first
    cache_dir = project_root / "cache"
    cached_data_file = cache_dir / "historical_data.csv"
    
    try:
        if cached_data_file.exists():
            print("📊 Loading cached historical data...")
            historical_data = pd.read_csv(cached_data_file)
            print(f"   ✅ Loaded {len(historical_data)} records from cache")
        else:
            print("📡 Loading fresh historical data from FastF1...")
            print("   ⏳ This may take 15-30 minutes for the first run...")
            
            # Load fresh data
            loader = F1DataLoader(cache_dir=str(cache_dir))
            historical_data = loader.load_historical_data(start_year=2020, end_year=2024)
            
            # Save for future use
            historical_data.to_csv(cached_data_file, index=False)
            print(f"   💾 Data cached for future runs")
        
        # Create and train the predictor
        print(f"\n🤖 Creating and training F1 prediction model...")
        predictor, metrics = create_f1_predictor(historical_data)
        
        # Store the engineered historical data for predictions
        engineered_historical_data = predictor.engineer_features(historical_data.copy())
        
        # Display training results
        print(f"\n📈 Model Performance:")
        for metric_name, value in metrics.items():
            if 'acc' in metric_name:
                print(f"   {metric_name}: {value:.3f}")
            elif 'rmse' in metric_name:
                print(f"   {metric_name}: {value:.2f}")
        
        # Get user input for race prediction
        print(f"\n🏁 Available prediction options:")
        print("1. Dutch Grand Prix 2025")
        print("2. Custom race prediction")
        print("3. All 2025 drivers analysis")
        
        while True:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                # Dutch GP prediction
                print(f"\n🇳🇱 Generating Dutch Grand Prix 2025 predictions...")
                predictions = predictor.predict_race(
                    race_name="Dutch Grand Prix",
                    driver_team_dict=DRIVER_TEAM_2025,
                    historical_data=historical_data
                )
                
                display_predictions(predictions, predictor, "Dutch Grand Prix 2025")
                
                # Show historical context
                dutch_history = historical_data[
                    historical_data['race'].str.contains('Dutch', case=False, na=False)
                ]
                print(f"\n📊 Analysis based on {len(historical_data)} historical records (2020-2024)")
                print(f"🇳🇱 Dutch GP historical data: {len(dutch_history)} records analyzed")
                break
                
            elif choice == "2":
                # Custom race
                race_name = input("Enter race name (e.g., 'Monaco Grand Prix'): ").strip()
                if race_name:
                    print(f"\n🏁 Generating {race_name} 2025 predictions...")
                    predictions = predictor.predict_race(
                        race_name=race_name,
                        driver_team_dict=DRIVER_TEAM_2025,
                        historical_data=historical_data
                    )
                    
                    display_predictions(predictions, predictor, f"{race_name} 2025")
                    print(f"\n📊 Analysis based on {len(historical_data)} historical records")
                    break
                else:
                    print("Please enter a valid race name.")
                    
            elif choice == "3":
                # Driver analysis
                print(f"\n👥 Analyzing all 2025 F1 drivers...")
                predictions = predictor.predict_race(
                    race_name="Generic Circuit",
                    driver_team_dict=DRIVER_TEAM_2025,
                    historical_data=historical_data
                )
                
                # Show driver rankings
                driver_summary = predictions.groupby('driver').agg({
                    'win_prob': 'max',
                    'podium_prob': 'max',
                    'team': 'first',
                    'is_rookie': 'first'
                }).sort_values('win_prob', ascending=False)
                
                print(f"\n🏆 2025 F1 DRIVER POWER RANKINGS")
                print("=" * 70)
                print(f"{'Rank':<5} {'Driver':<18} {'Team':<25} {'Max Win%':<8} {'Max Podium%':<10}")
                print("-" * 70)
                
                for i, (driver, stats) in enumerate(driver_summary.iterrows(), 1):
                    rookie_indicator = " 🆕" if stats.get('is_rookie', False) else ""
                    print(f"{i:<5} {driver:<18} {stats['team']:<25} "
                          f"{stats['win_prob']:<8.1f} {stats['podium_prob']:<10.1f}{rookie_indicator}")
                
                print(f"\n📊 Based on optimal starting positions across all scenarios")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure FastF1 is installed: pip install fastf1")
        print("2. Check your internet connection (required for data loading)")
        print("3. Ensure you have sufficient disk space for caching (~3GB)")
        print("4. Try running setup.py first to initialize the environment")
        return 1
    
    print(f"\n💡 Predictions powered by advanced ML with 2025 lineup updates!")
    print("🏁 Thank you for using F1 Prediction Suite!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
