#!/usr/bin/env python3
"""
F1 Prediction Suite - Setup Script
This script sets up the complete environment, installs dependencies, 
and loads historical F1 data for the prediction models.
"""

import subprocess
import sys
import os
import venv
from pathlib import Path


class F1PredictionSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.cache_path = self.project_root / "cache"
        self.src_path = self.project_root / "src"
        
    def create_virtual_environment(self):
        """Create a virtual environment for the project"""
        print("🔧 Creating virtual environment...")
        
        if self.venv_path.exists():
            print("   Virtual environment already exists")
            return True
            
        try:
            venv.create(self.venv_path, with_pip=True)
            print("   ✅ Virtual environment created successfully")
            return True
        except Exception as e:
            print(f"   ❌ Error creating virtual environment: {e}")
            return False
    
    def get_pip_executable(self):
        """Get the pip executable path for the virtual environment"""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def get_python_executable(self):
        """Get the Python executable path for the virtual environment"""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def install_dependencies(self):
        """Install required Python packages in the virtual environment"""
        print("📦 Installing dependencies...")
        
        pip_exe = self.get_pip_executable()
        
        # Core dependencies
        packages = [
            "pandas>=2.0.0",
            "numpy>=1.20.0", 
            "scikit-learn>=1.3.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "fastf1>=3.1.0",
            "plotly>=5.0.0",
            "requests>=2.25.0"
        ]
        
        for package in packages:
            try:
                print(f"   Installing {package}...")
                subprocess.check_call([str(pip_exe), "install", package], 
                                    capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}")
                print(f"      Error: {e.stderr if hasattr(e, 'stderr') else str(e)}")
                return False
        
        print("   ✅ All dependencies installed successfully!")
        return True
    
    def create_directory_structure(self):
        """Create the project directory structure"""
        print("📁 Creating project directory structure...")
        
        directories = [
            "src",
            "data",
            "models", 
            "cache",
            "examples",
            "tests",
            "docs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"   ✅ Created: {directory}/")
        
        return True
    
    def load_historical_data(self):
        """Load historical F1 data using the data loader"""
        print("📡 Loading historical F1 data (this may take a while)...")
        
        python_exe = self.get_python_executable()
        
        # Create a script to run the data loader
        data_loading_script = f'''
import sys
sys.path.append(r"{self.src_path}")

from data_loader import F1DataLoader
import os

# Create cache directory
cache_dir = r"{self.cache_path}"
os.makedirs(cache_dir, exist_ok=True)

# Load data
try:
    loader = F1DataLoader(cache_dir=cache_dir)
    data = loader.load_historical_data(start_year=2020, end_year=2024)
    
    # Save processed data for quick access
    data_file = os.path.join(cache_dir, "historical_data.csv")
    data.to_csv(data_file, index=False)
    
    cache_info = loader.get_cache_info()
    print(f"\\n✅ Data loading complete!")
    print(f"   📊 Cache size: {{cache_info['size_mb']}} MB")
    print(f"   📄 Files cached: {{cache_info['file_count']}}")
    print(f"   💾 Processed data saved to: {{data_file}}")
    
except Exception as e:
    print(f"❌ Error loading data: {{e}}")
    sys.exit(1)
'''
        
        try:
            result = subprocess.run([str(python_exe), "-c", data_loading_script], 
                                  capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print(result.stdout)
                return True
            else:
                print(f"❌ Error loading data:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Data loading timed out (1 hour limit)")
            return False
        except Exception as e:
            print(f"❌ Error running data loader: {e}")
            return False
    
    def create_activation_script(self):
        """Create a script to activate the virtual environment"""
        print("📜 Creating activation script...")
        
        if sys.platform == "win32":
            activate_script = self.project_root / "activate.bat"
            script_content = f'''@echo off
echo 🏎️ F1 Prediction Suite Environment
echo Activating virtual environment...
call "{self.venv_path}\\Scripts\\activate.bat"
echo ✅ Environment activated!
echo.
echo Available commands:
echo   python examples/run_predictions.py  - Run F1 predictions
echo   python src/data_loader.py           - Load fresh data
echo.
cmd /k
'''
        else:
            activate_script = self.project_root / "activate.sh"
            script_content = f'''#!/bin/bash
echo "🏎️ F1 Prediction Suite Environment"
echo "Activating virtual environment..."
source "{self.venv_path}/bin/activate"
echo "✅ Environment activated!"
echo ""
echo "Available commands:"
echo "  python examples/run_predictions.py  - Run F1 predictions"
echo "  python src/data_loader.py           - Load fresh data"
echo ""
bash
'''
        
        with open(activate_script, 'w') as f:
            f.write(script_content)
        
        if not sys.platform == "win32":
            os.chmod(activate_script, 0o755)
        
        print(f"   ✅ Created: {activate_script.name}")
        return True
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🏎️ F1 PREDICTION SUITE - SETUP")
        print("=" * 50)
        print("This will create a complete environment for F1 race predictions")
        print("Including virtual environment, dependencies, and 5+ years of F1 data")
        print("=" * 50)
        
        # Confirm before proceeding
        confirm = input("\nProceed with setup? This may take 30-60 minutes. (y/N): ")
        if confirm.lower() not in ['y', 'yes']:
            print("Setup cancelled.")
            return False
        
        steps = [
            ("Creating directory structure", self.create_directory_structure),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Loading historical F1 data", self.load_historical_data),
            ("Creating activation script", self.create_activation_script)
        ]
        
        for step_name, step_function in steps:
            print(f"\n{step_name}...")
            if not step_function():
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        print("\n" + "=" * 50)
        print("🎉 F1 PREDICTION SUITE SETUP COMPLETE!")
        print("=" * 50)
        print(f"📁 Project root: {self.project_root}")
        print(f"🐍 Virtual environment: {self.venv_path}")
        print(f"💾 Data cache: {self.cache_path}")
        print("")
        print("To get started:")
        
        if sys.platform == "win32":
            print("  1. Run: activate.bat")
        else:
            print("  1. Run: ./activate.sh")
        
        print("  2. Run: python examples/run_predictions.py")
        print("")
        print("Happy predicting! 🏁")
        return True


def main():
    """Main setup function"""
    setup = F1PredictionSetup()
    success = setup.run_setup()
    
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
