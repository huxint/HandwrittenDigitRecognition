"""
Main entry point for the Handwritten Digit Recognition Application
"""

import sys
import os
import argparse

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Handwritten Digit Recognition Application')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--gui', action='store_true', default=True, help='Launch GUI application (default)')
    parser.add_argument('--model-path', type=str, default='models/digit_model.pth', 
                       help='Path to model file')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training a new model...")
        from src.model.trainer import train_model
        train_model()
    else:
        print("Launching GUI application...")
        from src.gui.drawing_app import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
