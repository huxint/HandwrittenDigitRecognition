"""
Image processing utilities for digit recognition
"""

import numpy as np
import cv2
from PIL import Image, ImageOps
import torch
from torchvision import transforms


class ImageProcessor:
    """
    Image processing utilities for preparing drawn digits for model prediction
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_canvas_image(self, image_array, canvas_size=(280, 280), target_size=(28, 28)):
        """
        Preprocess image from drawing canvas for model prediction
        
        Args:
            image_array: numpy array of the drawn image
            canvas_size: size of the drawing canvas
            target_size: target size for model input (28x28 for MNIST)
            
        Returns:
            torch.Tensor: preprocessed image tensor ready for model
        """
        # Convert to PIL Image if numpy array
        if isinstance(image_array, np.ndarray):
            # Ensure it's in the right format (0-255, uint8)
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Convert RGB to grayscale if needed
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            image = Image.fromarray(image_array)
        else:
            image = image_array
        
        # Convert to grayscale and invert colors (white background, black text -> black background, white text)
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image)
        
        # Find bounding box of the drawn content
        bbox = self.get_digit_bbox(np.array(image))
        
        if bbox is not None:
            # Crop to bounding box with some padding
            image = self.crop_and_pad(image, bbox)
        
        # Resize to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Apply normalization transform
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def get_digit_bbox(self, image_array, threshold=30):
        """
        Get bounding box of the drawn digit
        
        Args:
            image_array: grayscale image array
            threshold: threshold for detecting drawn content
            
        Returns:
            tuple: (left, top, right, bottom) or None if no content found
        """
        # Find pixels above threshold
        rows = np.any(image_array > threshold, axis=1)
        cols = np.any(image_array > threshold, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return (cmin, rmin, cmax, rmax)
    
    def crop_and_pad(self, image, bbox, padding_ratio=0.1):
        """
        Crop image to bounding box and add padding
        
        Args:
            image: PIL Image
            bbox: bounding box (left, top, right, bottom)
            padding_ratio: ratio of padding to add
            
        Returns:
            PIL Image: cropped and padded image
        """
        left, top, right, bottom = bbox
        
        # Calculate padding
        width = right - left
        height = bottom - top
        pad_x = int(width * padding_ratio)
        pad_y = int(height * padding_ratio)
        
        # Expand bounding box with padding
        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(image.width, right + pad_x)
        bottom = min(image.height, bottom + pad_y)
        
        # Crop image
        cropped = image.crop((left, top, right, bottom))
        
        # Make it square by adding padding to the shorter dimension
        width, height = cropped.size
        if width > height:
            # Add padding to height
            pad_height = (width - height) // 2
            cropped = ImageOps.expand(cropped, (0, pad_height, 0, width - height - pad_height), fill=0)
        elif height > width:
            # Add padding to width
            pad_width = (height - width) // 2
            cropped = ImageOps.expand(cropped, (pad_width, 0, height - width - pad_width, 0), fill=0)
        
        return cropped
    
    def canvas_to_array(self, canvas_widget):
        """
        Convert tkinter canvas to numpy array
        
        Args:
            canvas_widget: tkinter Canvas widget
            
        Returns:
            numpy array: image array
        """
        # This is a placeholder - actual implementation depends on how we capture canvas content
        # We'll implement this in the GUI module
        pass
    
    def enhance_image(self, image_array):
        """
        Apply image enhancement techniques
        
        Args:
            image_array: input image array
            
        Returns:
            numpy array: enhanced image
        """
        # Apply Gaussian blur to smooth the image
        blurred = cv2.GaussianBlur(image_array, (3, 3), 0)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def visualize_preprocessing(self, original, processed):
        """
        Visualize the preprocessing steps (for debugging)
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Original')
        ax1.axis('off')
        
        # Convert tensor back to numpy for visualization
        if isinstance(processed, torch.Tensor):
            processed_np = processed.squeeze().numpy()
        else:
            processed_np = processed
            
        ax2.imshow(processed_np, cmap='gray')
        ax2.set_title('Processed')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()


# Utility functions
def tensor_to_image(tensor):
    """Convert tensor back to PIL Image for visualization"""
    # Remove batch dimension and convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    
    # Denormalize
    tensor = tensor * 0.3081 + 0.1307
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    image_array = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_array)


def save_debug_image(image, filename):
    """Save image for debugging purposes"""
    if isinstance(image, torch.Tensor):
        image = tensor_to_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image.save(filename)
    print(f"Debug image saved: {filename}")
