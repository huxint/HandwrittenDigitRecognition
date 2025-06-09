"""
Demo script to test the digit recognition model with sample images
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.digit_cnn import SimpleCNN, DigitCNN
from src.utils.image_processing import ImageProcessor


def load_model(model_path='models/digit_model.pth'):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first by running: python train_model.py")
        return None, None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_class = checkpoint.get('model_class', 'SimpleCNN')
        
        if model_class == 'SimpleCNN':
            model = SimpleCNN().to(device)
        else:
            model = DigitCNN().to(device)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded: {model_class} on {device}")
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def test_with_mnist_samples(model, device, num_samples=10):
    """Test the model with random MNIST samples"""
    print(f"\nTesting with {num_samples} random MNIST samples...")
    
    # Load MNIST test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Get random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    correct = 0
    
    # Create subplot for visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        image, true_label = test_dataset[idx]
        
        # Add batch dimension
        image_batch = image.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            probabilities = model.predict_proba(image_batch)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_label].item() * 100
        
        # Check if correct
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
        
        # Plot
        axes[i].imshow(image.squeeze(), cmap='gray')
        color = 'green' if is_correct else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {predicted_label}\nConf: {confidence:.1f}%', 
                         color=color, fontsize=10)
        axes[i].axis('off')
        
        print(f"Sample {i+1}: True={true_label}, Predicted={predicted_label}, "
              f"Confidence={confidence:.1f}%, {'✓' if is_correct else '✗'}")
    
    accuracy = correct / num_samples * 100
    print(f"\nAccuracy on {num_samples} samples: {accuracy:.1f}% ({correct}/{num_samples})")
    
    plt.tight_layout()
    plt.suptitle(f'MNIST Test Samples - Accuracy: {accuracy:.1f}%', fontsize=14)
    plt.show()
    
    return accuracy


def create_synthetic_digit(digit, size=(28, 28)):
    """Create a simple synthetic digit for testing"""
    image = np.zeros(size)
    
    if digit == 0:
        # Draw a circle
        center = (size[0]//2, size[1]//2)
        radius = min(size)//3
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        outer_mask = (x - center[1])**2 + (y - center[0])**2 <= (radius-3)**2
        image[mask & ~outer_mask] = 1.0
        
    elif digit == 1:
        # Draw a vertical line
        center_x = size[1]//2
        image[4:-4, center_x-1:center_x+2] = 1.0
        
    # Add more digits as needed...
    
    return image


def test_with_synthetic_digits(model, device):
    """Test with simple synthetic digits"""
    print("\nTesting with synthetic digits...")
    
    processor = ImageProcessor()
    
    for digit in [0, 1]:
        print(f"\nTesting synthetic digit {digit}:")
        
        # Create synthetic image
        synthetic_image = create_synthetic_digit(digit)
        
        # Convert to tensor (skip normalization for synthetic)
        tensor = torch.FloatTensor(synthetic_image).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            probabilities = model.predict_proba(tensor)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_label].item() * 100
        
        print(f"Predicted: {predicted_label}, Confidence: {confidence:.1f}%")
        
        # Show top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        print("Top 3 predictions:")
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            print(f"  {i+1}. Digit {idx.item()}: {prob.item()*100:.1f}%")


def main():
    """Main demo function"""
    print("=" * 50)
    print("Handwritten Digit Recognition - Demo")
    print("=" * 50)
    
    # Load model
    model, device = load_model()
    if model is None:
        return
    
    # Test with MNIST samples
    try:
        accuracy = test_with_mnist_samples(model, device, num_samples=10)
    except Exception as e:
        print(f"Error testing with MNIST: {e}")
        accuracy = None
    
    # Test with synthetic digits
    try:
        test_with_synthetic_digits(model, device)
    except Exception as e:
        print(f"Error testing with synthetic digits: {e}")
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    if accuracy is not None:
        print(f"Model accuracy on test samples: {accuracy:.1f}%")
    print("To use the interactive GUI, run: python main.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
