"""
Test script to verify the installation and basic functionality
"""

import sys
import os
import importlib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        'torch',
        'torchvision', 
        'numpy',
        'PIL',
        'matplotlib',
        'cv2',
        'sklearn',
        'tqdm'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0, failed_imports


def test_project_modules():
    """Test if project modules can be imported"""
    print("\nTesting project modules...")
    
    project_modules = [
        'src.model.digit_cnn',
        'src.model.trainer',
        'src.utils.image_processing',
        'src.gui.drawing_app'
    ]
    
    failed_imports = []
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0, failed_imports


def test_model_creation():
    """Test if models can be created"""
    print("\nTesting model creation...")
    
    try:
        from src.model.digit_cnn import SimpleCNN, DigitCNN
        
        # Test SimpleCNN
        simple_model = SimpleCNN()
        print(f"✓ SimpleCNN created - Parameters: {sum(p.numel() for p in simple_model.parameters())}")
        
        # Test DigitCNN
        digit_model = DigitCNN()
        print(f"✓ DigitCNN created - Parameters: {sum(p.numel() for p in digit_model.parameters())}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_image_processing():
    """Test image processing functionality"""
    print("\nTesting image processing...")
    
    try:
        from src.utils.image_processing import ImageProcessor
        import numpy as np
        
        processor = ImageProcessor()
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (280, 280, 3), dtype=np.uint8)
        
        # Test preprocessing
        tensor = processor.preprocess_canvas_image(dummy_image)
        print(f"✓ Image preprocessing - Output shape: {tensor.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        return False


def test_directories():
    """Test if required directories exist"""
    print("\nTesting directories...")
    
    required_dirs = ['src', 'src/model', 'src/gui', 'src/utils']
    optional_dirs = ['models', 'data']
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} (missing)")
            all_good = False
    
    for directory in optional_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"⚠ {directory} (will be created when needed)")
    
    return all_good


def main():
    """Run all tests"""
    print("=" * 50)
    print("Handwritten Digit Recognition - Installation Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test directories
    if not test_directories():
        all_tests_passed = False
    
    # Test imports
    imports_ok, failed_imports = test_imports()
    if not imports_ok:
        all_tests_passed = False
        print(f"\nFailed imports: {failed_imports}")
        print("Please install missing packages with: pip install -r requirements.txt")
    
    # Test project modules
    if imports_ok:  # Only test if basic imports work
        project_ok, failed_project = test_project_modules()
        if not project_ok:
            all_tests_passed = False
            print(f"\nFailed project imports: {failed_project}")
    
        # Test model creation
        if project_ok:
            if not test_model_creation():
                all_tests_passed = False
        
            # Test image processing
            if not test_image_processing():
                all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All tests passed! Installation looks good.")
        print("\nYou can now:")
        print("1. Train a model: python train_model.py")
        print("2. Run the GUI: python main.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTry running: python setup.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
