"""
Setup script for the Handwritten Digit Recognition project
"""

import os
import subprocess
import sys


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    directories = ['models', 'data']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required!")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True


def train_initial_model():
    """Train the initial model"""
    print("\nWould you like to train the initial model now?")
    print("This will download MNIST data (~50MB) and train a model (~5-10 minutes)")
    
    response = input("Train model now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("Training model...")
        try:
            import train_model
            train_model.main()
            print("✓ Model training completed!")
            return True
        except Exception as e:
            print(f"✗ Error training model: {e}")
            print("You can train the model later by running: python train_model.py")
            return False
    else:
        print("Skipping model training. You can train later with: python train_model.py")
        return True


def main():
    """Main setup function"""
    print("=" * 60)
    print("Handwritten Digit Recognition - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\nSetup failed. Please install requirements manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Train initial model
    train_initial_model()
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nTo run the application:")
    print("  python main.py")
    print("\nTo train a new model:")
    print("  python train_model.py")
    print("\nFor help:")
    print("  python main.py --help")
    print("\nEnjoy recognizing digits!")


if __name__ == "__main__":
    main()
