#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECG Anesthesia Delirium Prediction System - Streamlit Launch Script

Provides convenient launch options and system check functionality
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def check_dependencies():
    """Check required dependencies"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies checked")
    return True

def check_models():
    """Check model files"""
    model_dirs = [
        'delirium_model_results/models',
        'models',
        'output/models'
    ]
    
    found_models = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if model_files:
                found_models.extend([os.path.join(model_dir, f) for f in model_files])
    
    if found_models:
        print("✅ Found model files:")
        for model in found_models:
            print(f"   📄 {model}")
        return True
    else:
        print("⚠️  No model files found")
        print("Please train models first:")
        print("python run_delirium_system.py --mode train --delirium data/PND_long_sequences --non_delirium data/NPND_long_sequences")
        return False

def check_data_format():
    """Check sample data format"""
    sample_dirs = [
        'data/NPND_long_sequences',
        'data/PND_long_sequences'
    ]
    
    for data_dir in sample_dirs:
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                print(f"✅ Found sample data: {data_dir}")
                return True
    
    print("⚠️  No sample data files found")
    return False

def setup_streamlit_config():
    """Setup Streamlit configuration"""
    config_dir = Path.home() / '.streamlit'
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / 'config.toml'
    
    if not config_file.exists():
        # Copy config file
        if os.path.exists('streamlit_config.toml'):
            shutil.copy('streamlit_config.toml', config_file)
            print(f"✅ Streamlit config set: {config_file}")
        else:
            print("⚠️  streamlit_config.toml not found")

def run_streamlit_app(port=8501, host='localhost', debug=False):
    """Launch Streamlit application"""
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found")
        return False
    
    # Build launch command
    cmd = [
        'streamlit', 'run', 'app.py',
        '--server.port', str(port),
        '--server.address', host
    ]
    
    if debug:
        cmd.extend(['--logger.level', 'debug'])
    
    print(f"🚀 Launching Streamlit application...")
    print(f"   Address: http://{host}:{port}")
    print(f"   Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop the application\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Launch failed: {str(e)}")
        return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ECG Anesthesia Delirium Prediction System Launcher')
    
    parser.add_argument('--check', action='store_true',
                        help='Check system environment and dependencies')
    
    parser.add_argument('--port', type=int, default=8501,
                        help='Specify port number (default: 8501)')
    
    parser.add_argument('--host', type=str, default='localhost',
                        help='Specify host address (default: localhost)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    parser.add_argument('--setup-config', action='store_true',
                        help='Setup Streamlit config file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏥 ECG Anesthesia Delirium Prediction System - Web Launcher")
    print("=" * 60)
    
    # Check system environment
    if args.check:
        print("\n🔍 System Environment Check:")
        print("-" * 40)
        
        print("\n1. Checking Python dependencies:")
        deps_ok = check_dependencies()
        
        print("\n2. Checking model files:")
        models_ok = check_models()
        
        print("\n3. Checking data files:")
        data_ok = check_data_format()
        
        print("\n📋 Check Summary:")
        print(f"   Dependencies: {'✅' if deps_ok else '❌'}")
        print(f"   Model files: {'✅' if models_ok else '⚠️'}")
        print(f"   Sample data: {'✅' if data_ok else '⚠️'}")
        
        if not deps_ok:
            print("\n❌ Missing dependencies, cannot launch application")
            return 1
        
        if not models_ok:
            print("\n⚠️  Recommend training models for best experience")
        
        print("\n✅ System check complete")
        return 0
    
    # Setup config file
    if args.setup_config:
        setup_streamlit_config()
        return 0
    
    # Quick check for critical dependencies
    try:
        import streamlit
        print("✅ Streamlit installed")
    except ImportError:
        print("❌ Streamlit not installed, please run: pip install streamlit")
        return 1
    
    # Setup configuration
    setup_streamlit_config()
    
    # Launch application
    success = run_streamlit_app(
        port=args.port,
        host=args.host,
        debug=args.debug
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
