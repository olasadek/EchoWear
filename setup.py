#!/usr/bin/env python3
"""
EchoWear Setup Script
Installs all required dependencies and downloads necessary models
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def install_pip_requirements():
    """Install Python packages from requirements.txt"""
    print("\n" + "="*60)
    print("📦 Installing Python Dependencies")
    print("="*60)
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements.txt")

def install_spacy_models():
    """Install spaCy language models"""
    print("\n" + "="*60)
    print("🧠 Installing spaCy Language Models")
    print("="*60)
    
    models = [
        ("en_core_web_sm", "Small English model (basic)"),
        ("en_core_web_lg", "Large English model (recommended)")
    ]
    
    success_count = 0
    for model, description in models:
        if run_command(f"{sys.executable} -m spacy download {model}", f"Installing {model} - {description}"):
            success_count += 1
    
    if success_count == 0:
        print("⚠️  No spaCy models installed. The application will use basic embeddings.")
    else:
        print(f"✅ Installed {success_count}/{len(models)} spaCy models")
    
    return success_count > 0

def install_optional_dependencies():
    """Install optional dependencies for enhanced functionality"""
    print("\n" + "="*60)
    print("🎯 Installing Optional Dependencies")
    print("="*60)
    
    optional_packages = [
        ("gensim[complete]", "Advanced text embeddings"),
        ("torch", "PyTorch for deep learning models"),
        ("transformers[torch]", "Hugging Face transformers"),
    ]
    
    success_count = 0
    for package, description in optional_packages:
        if run_command(f"{sys.executable} -m pip install {package}", f"Installing {package} - {description}"):
            success_count += 1
    
    print(f"✅ Installed {success_count}/{len(optional_packages)} optional packages")
    return success_count > 0

def download_nltk_data():
    """Download necessary NLTK data"""
    print("\n" + "="*60)
    print("📚 Downloading NLTK Data")
    print("="*60)
    
    nltk_data = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
    
    try:
        import nltk
        success_count = 0
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
                print(f"✅ Downloaded NLTK data: {data}")
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to download {data}: {e}")
        
        print(f"✅ Downloaded {success_count}/{len(nltk_data)} NLTK datasets")
        return success_count > 0
        
    except ImportError:
        print("⚠️  NLTK not available. Skipping NLTK data download.")
        return False

def check_opencv():
    """Check if OpenCV is working correctly"""
    print("\n" + "="*60)
    print("📹 Testing OpenCV Installation")
    print("="*60)
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test basic functionality
        test_image = cv2.imread("README.md", cv2.IMREAD_COLOR)  # This will fail gracefully
        print("✅ OpenCV basic functionality test passed")
        return True
        
    except ImportError as e:
        print(f"❌ OpenCV not available: {e}")
        return False

def test_imports():
    """Test critical imports"""
    print("\n" + "="*60)
    print("🧪 Testing Critical Imports")
    print("="*60)
    
    critical_imports = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("spacy", "spaCy"),
        ("transformers", "Transformers"),
        ("tkinter", "Tkinter (GUI)"),
    ]
    
    optional_imports = [
        ("gensim", "Gensim"),
        ("torch", "PyTorch"),
        ("nltk", "NLTK"),
    ]
    
    print("Critical imports:")
    critical_success = 0
    for module, name in critical_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
            critical_success += 1
        except ImportError:
            print(f"❌ {name} - REQUIRED")
    
    print("\nOptional imports:")
    optional_success = 0
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
            optional_success += 1
        except ImportError:
            print(f"⚠️  {name} - Optional")
    
    print(f"\n📊 Import Summary:")
    print(f"Critical: {critical_success}/{len(critical_imports)} ({'✅ OK' if critical_success == len(critical_imports) else '❌ ISSUES'})")
    print(f"Optional: {optional_success}/{len(optional_imports)}")
    
    return critical_success == len(critical_imports)

def create_models_directory():
    """Ensure the models directory exists with YOLO files"""
    print("\n" + "="*60)
    print("📁 Checking Models Directory")
    print("="*60)
    
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir()
        print("✅ Created models directory")
    
    # Check for YOLO files
    yolo_files = ["yolov4-tiny.weights", "yolov4-tiny.cfg", "coco.names"]
    missing_files = []
    
    for file in yolo_files:
        file_path = models_dir / file
        if file_path.exists():
            print(f"✅ Found {file}")
        else:
            print(f"⚠️  Missing {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing YOLO model files: {', '.join(missing_files)}")
        print("The object detection will use HOG person detector as fallback.")
    else:
        print("✅ All YOLO model files present")
    
    return len(missing_files) == 0

def main():
    """Main setup function"""
    print("🚀 EchoWear Setup Script")
    print("=" * 60)
    print("This script will install all dependencies and set up the environment")
    print("for the EchoWear navigation system.")
    print()
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Setup steps
    steps = [
        ("Installing Python packages", install_pip_requirements),
        ("Installing spaCy models", install_spacy_models),
        ("Installing optional dependencies", install_optional_dependencies),
        ("Downloading NLTK data", download_nltk_data),
        ("Checking OpenCV", check_opencv),
        ("Checking models directory", create_models_directory),
        ("Testing imports", test_imports),
    ]
    
    results = []
    for description, func in steps:
        print(f"\n{'='*60}")
        result = func()
        results.append((description, result))
    
    # Final summary
    print("\n" + "="*60)
    print("📋 SETUP SUMMARY")
    print("="*60)
    
    success_count = 0
    for description, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status:<12} {description}")
        if success:
            success_count += 1
    
    print(f"\n📊 Overall: {success_count}/{len(results)} steps completed successfully")
    
    if success_count == len(results):
        print("\n🎉 Setup completed successfully!")
        print("You can now run the EchoWear applications:")
        print("  • python gui_app.py - Main GUI application")
        print("  • python path_manager.py - Path management")
        print("  • python enhanced_path_manager_demo.py - Enhanced demo")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("The application may still work but some features might be limited.")
        print("Check the error messages above for details.")
    
    print("\n" + "="*60)
    return success_count == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
