"""
GUI Application for Handwritten Digit Recognition
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model.digit_cnn import SimpleCNN, DigitCNN
from src.utils.image_processing import ImageProcessor


class DigitRecognitionApp:
    """
    Main application class for handwritten digit recognition
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.canvas_size = 280
        self.brush_size = 15
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Initialize model and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = ImageProcessor()
        
        # PIL Image for drawing
        self.image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Setup GUI
        self.setup_gui()
        
        # Load model
        self.load_model()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Drawing canvas
        canvas_frame = ttk.LabelFrame(main_frame, text="Draw a digit (0-9)", padding="10")
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            cursor='pencil'
        )
        self.canvas.pack(pady=10)
        
        # Canvas bindings
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_on_canvas)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Canvas controls
        canvas_controls = ttk.Frame(canvas_frame)
        canvas_controls.pack(pady=10)
        
        ttk.Button(canvas_controls, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(canvas_controls, text="Predict", command=self.predict_digit).pack(side=tk.LEFT, padx=5)
        
        # Brush size control
        brush_frame = ttk.Frame(canvas_frame)
        brush_frame.pack(pady=5)
        ttk.Label(brush_frame, text="Brush Size:").pack(side=tk.LEFT)
        self.brush_scale = ttk.Scale(
            brush_frame,
            from_=5,
            to=25,
            orient=tk.HORIZONTAL,
            length=150,
            command=self.update_brush_size
        )
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Predictions
        prediction_frame = ttk.LabelFrame(main_frame, text="Predictions", padding="10")
        prediction_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Main prediction display
        self.prediction_label = ttk.Label(
            prediction_frame,
            text="Draw a digit to see prediction",
            font=('Arial', 24, 'bold'),
            foreground='blue'
        )
        self.prediction_label.pack(pady=20)
        
        # Confidence display
        self.confidence_label = ttk.Label(
            prediction_frame,
            text="Confidence: --",
            font=('Arial', 14)
        )
        self.confidence_label.pack(pady=10)
        
        # Top 3 predictions
        top3_frame = ttk.LabelFrame(prediction_frame, text="Top 3 Predictions", padding="10")
        top3_frame.pack(pady=20, fill=tk.X)
        
        self.top3_labels = []
        for i in range(3):
            label = ttk.Label(top3_frame, text=f"{i+1}. -- (---%)", font=('Arial', 12))
            label.pack(anchor=tk.W, pady=2)
            self.top3_labels.append(label)
        
        # Model info
        model_frame = ttk.LabelFrame(prediction_frame, text="Model Information", padding="10")
        model_frame.pack(pady=20, fill=tk.X)
        
        self.model_info_label = ttk.Label(
            model_frame,
            text="Model: Not loaded",
            font=('Arial', 10)
        )
        self.model_info_label.pack(anchor=tk.W)
        
        self.device_label = ttk.Label(
            model_frame,
            text=f"Device: {self.device}",
            font=('Arial', 10)
        )
        self.device_label.pack(anchor=tk.W)
        
        # Menu bar
        self.setup_menu()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model...", command=self.load_model_dialog)
        file_menu.add_command(label="Save Drawing...", command=self.save_drawing)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Train New Model", command=self.train_model_dialog)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def start_drawing(self, event):
        """Start drawing on canvas"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_on_canvas(self, event):
        """Draw on canvas"""
        if self.drawing:
            # Draw on tkinter canvas
            self.canvas.create_oval(
                event.x - self.brush_size//2,
                event.y - self.brush_size//2,
                event.x + self.brush_size//2,
                event.y + self.brush_size//2,
                fill='black',
                outline='black'
            )
            
            # Draw on PIL image
            self.draw.ellipse([
                event.x - self.brush_size//2,
                event.y - self.brush_size//2,
                event.x + self.brush_size//2,
                event.y + self.brush_size//2
            ], fill='black')
            
            self.last_x = event.x
            self.last_y = event.y
            
            # Auto-predict if model is loaded
            if self.model is not None:
                self.root.after(100, self.predict_digit)  # Delay prediction slightly
                
    def stop_drawing(self, event):
        """Stop drawing"""
        self.drawing = False
        
    def update_brush_size(self, value):
        """Update brush size"""
        self.brush_size = int(float(value))
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Clear predictions
        self.prediction_label.config(text="Draw a digit to see prediction")
        self.confidence_label.config(text="Confidence: --")
        for label in self.top3_labels:
            label.config(text="-- (---%)")
            
        self.status_var.set("Canvas cleared")
        
    def predict_digit(self):
        """Predict the drawn digit"""
        if self.model is None:
            self.status_var.set("No model loaded")
            return
            
        try:
            # Convert PIL image to tensor
            tensor = self.processor.preprocess_canvas_image(np.array(self.image))
            tensor = tensor.to(self.device)
            
            # Get predictions
            probabilities = self.model.predict_proba(tensor)
            probs = probabilities.cpu().numpy()[0]
            
            # Get top prediction
            predicted_digit = np.argmax(probs)
            confidence = probs[predicted_digit] * 100
            
            # Update main prediction
            self.prediction_label.config(text=f"Predicted: {predicted_digit}")
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            
            # Update top 3 predictions
            top3_indices = np.argsort(probs)[-3:][::-1]
            for i, idx in enumerate(top3_indices):
                self.top3_labels[i].config(
                    text=f"{i+1}. Digit {idx} ({probs[idx]*100:.1f}%)"
                )
            
            self.status_var.set(f"Predicted: {predicted_digit} ({confidence:.1f}%)")
            
        except Exception as e:
            self.status_var.set(f"Prediction error: {str(e)}")
            
    def load_model(self, model_path=None):
        """Load the trained model"""
        if model_path is None:
            model_path = os.path.join('models', 'digit_model.pth')
            
        try:
            if os.path.exists(model_path):
                # Load model
                checkpoint = torch.load(model_path, map_location=self.device)
                model_class = checkpoint.get('model_class', 'SimpleCNN')
                
                if model_class == 'SimpleCNN':
                    self.model = SimpleCNN().to(self.device)
                else:
                    self.model = DigitCNN().to(self.device)
                    
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                self.model_info_label.config(text=f"Model: {model_class} (Loaded)")
                self.status_var.set(f"Model loaded: {model_class}")
                
            else:
                self.model_info_label.config(text="Model: Not found - Train a model first")
                self.status_var.set("No trained model found")
                
        except Exception as e:
            self.model_info_label.config(text=f"Model: Error loading - {str(e)}")
            self.status_var.set(f"Error loading model: {str(e)}")
            
    def load_model_dialog(self):
        """Open dialog to load model"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            self.load_model(filename)
            
    def save_drawing(self):
        """Save the current drawing"""
        filename = filedialog.asksaveasfilename(
            title="Save Drawing",
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        if filename:
            self.image.save(filename)
            self.status_var.set(f"Drawing saved: {filename}")
            
    def train_model_dialog(self):
        """Show dialog for training a new model"""
        messagebox.showinfo(
            "Train Model",
            "To train a new model, run:\npython -m src.model.trainer\n\nThis will download MNIST data and train a model."
        )
        
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Handwritten Digit Recognition\n\n"
            "A deep learning application for recognizing handwritten digits (0-9).\n"
            "Built with PyTorch and Tkinter.\n\n"
            "Draw a digit on the canvas and see real-time predictions!"
        )


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
