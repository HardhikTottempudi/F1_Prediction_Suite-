"""
F1 Data Loader Module
Handles FastF1 data fetching and caching for F1 prediction models
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False


class F1DataLoader:
    """
    F1 Data Loader using FastF1 API
    Handles data fetching, caching, and preprocessing for machine learning models
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the F1 Data Loader
        
        Args:
            cache_dir (str): Directory to store FastF1 cache data
        """
        self.cache_dir = cache_dir
        
        if not FASTF1_AVAILABLE:
            raise ImportError("FastF1 not installed. Run: pip install fastf1")
        
        # Enable FastF1 caching
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        
        print(f"🏎️ F1 Data Loader initialized with cache directory: {cache_dir}")
    
    def load_historical_data(self, start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
        """
        Load historical F1 data for the specified year range
        
        Args:
            start_year (int): Starting year for data collection
            end_year (int): Ending year for data collection (inclusive)
            
        Returns:
            pd.DataFrame: Combined historical data with driver and race results
        """
        print(f"📡 Loading F1 historical data ({start_year}-{end_year})...")
        print("⏳ This may take a while for the first run (data will be cached)...")
        
        all_data = []
        
        for year in range(start_year, end_year + 1):
            print(f"\n📅 Processing {year} season...")
            
            try:
                schedule = fastf1.get_event_schedule(year)
                race_count = 0
                
                for _, event in schedule.iterrows():
                    # Skip non-conventional race formats (like sprint-only weekends)
                    if event['EventFormat'] != 'conventional' or pd.isna(event['Session5Date']):
                        continue
                    
                    event_name = event['EventName']
                    round_number = event['RoundNumber']
                    
                    try:
                        # Load qualifying and race sessions
                        qualifying = fastf1.get_session(year, round_number, 'Q')
                        race = fastf1.get_session(year, round_number, 'R')
                        
                        # Load with minimal data for faster processing
                        qualifying.load(laps=False, weather=False, messages=False, telemetry=False)
                        race.load(laps=False, weather=False, messages=False, telemetry=False)
                        
                        # Process each driver's qualifying and race results
                        for _, q_driver in qualifying.results.iterrows():
                            if pd.notna(q_driver['Position']):
                                driver_name = q_driver['FullName']
                                
                                # Find corresponding race result
                                race_result = race.results[race.results['FullName'] == driver_name]
                                
                                if not race_result.empty:
                                    race_driver = race_result.iloc[0]
                                    finish_pos = int(race_driver['Position']) if pd.notna(race_driver['Position']) else 21
                                    points = float(race_driver['Points']) if pd.notna(race_driver['Points']) else 0.0
                                    
                                    all_data.append({
                                        'year': year,
                                        'race': event_name,
                                        'driver': driver_name,
                                        'team': q_driver['TeamName'],
                                        'quali_pos': int(q_driver['Position']),
                                        'finish_pos': finish_pos,
                                        'points': points,
                                        'won': 1 if finish_pos == 1 else 0,
                                        'podium': 1 if finish_pos <= 3 else 0
                                    })
                        
                        race_count += 1
                        if race_count % 5 == 0:
                            print(f"    ✅ Processed {race_count} races...")
                            
                    except Exception as e:
                        print(f"    ⚠️ Error processing {event_name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ Error processing {year}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        
        if len(df) < 50:
            raise ValueError("Not enough historical data collected")
        
        # Sort chronologically for proper feature engineering
        df = df.sort_values(by=['year', 'race'], kind='mergesort').reset_index(drop=True)
        
        print(f"\n✅ Historical dataset loaded successfully!")
        print(f"   📊 {len(df)} driver records from {df['year'].min()}-{df['year'].max()}")
        print(f"   🏁 {df['race'].nunique()} unique races")
        print(f"   🏎️ {df['driver'].nunique()} unique drivers")
        
        return df
    
    def get_cache_info(self) -> dict:
        """
        Get information about the current cache status
        
        Returns:
            dict: Cache information including size and file count
        """
        if not os.path.exists(self.cache_dir):
            return {"cache_exists": False, "size_mb": 0, "file_count": 0}
        
        total_size = 0
        file_count = 0
        
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
                file_count += 1
        
        size_mb = total_size / (1024 * 1024)  # Convert to MB
        
        return {
            "cache_exists": True,
            "size_mb": round(size_mb, 2),
            "file_count": file_count,
            "cache_dir": self.cache_dir
        }
    
    def clear_cache(self) -> bool:
        """
        Clear the FastF1 cache directory
        
        Returns:
            bool: True if cache was cleared successfully
        """
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                print(f"✅ Cache cleared: {self.cache_dir}")
                return True
            return True
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False


def load_f1_data(start_year: int = 2020, end_year: int = 2024, cache_dir: str = "cache") -> pd.DataFrame:
    """
    Convenience function to load F1 historical data
    
    Args:
        start_year (int): Starting year for data collection
        end_year (int): Ending year for data collection (inclusive)
        cache_dir (str): Directory to store cache data
        
    Returns:
        pd.DataFrame: Historical F1 data
    """
    loader = F1DataLoader(cache_dir=cache_dir)
    return loader.load_historical_data(start_year=start_year, end_year=end_year)


if __name__ == "__main__":
    # Example usage
    print("🏎️ F1 Data Loader - Standalone Execution")
    print("=" * 50)
    
    try:
        # Load last 5 years of data
        data = load_f1_data(start_year=2020, end_year=2024)
        print(f"\nData shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Show sample data
        print("\nSample data:")
        print(data.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
