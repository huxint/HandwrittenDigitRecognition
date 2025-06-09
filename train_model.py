"""
Standalone script to train the digit recognition model
"""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.trainer import DigitTrainer


def main():
    """Train the digit recognition model"""
    print("=" * 50)
    print("Handwritten Digit Recognition - Model Training")
    print("=" * 50)
    
    # Create trainer
    print("Initializing trainer...")
    trainer = DigitTrainer(model_type='simple')
    
    # Prepare data
    print("Preparing MNIST dataset...")
    trainer.prepare_data(batch_size=64)
    
    # Train model
    print("Starting training...")
    trainer.train(epochs=10, save_path='models/digit_model.pth')
    
    # Plot training history
    print("Displaying training history...")
    try:
        trainer.plot_training_history()
    except Exception as e:
        print(f"Could not display plots: {e}")
    
    print("\nTraining completed!")
    print("Model saved to: models/digit_model.pth")
    print("You can now run the GUI application with: python main.py")


if __name__ == "__main__":
    main()
