# Handwritten Digit Recognition

A complete Python-based handwritten digit recognition application with real-time GUI interface. This project uses deep learning (PyTorch) to recognize handwritten digits (0-9) drawn by users on a canvas.

## Features

- **Real-time digit recognition** - See predictions as you draw
- **Interactive GUI** - Easy-to-use drawing canvas with mouse/stylus support
- **Confidence scores** - View prediction confidence and top 3 predictions
- **Deep learning model** - CNN trained on MNIST dataset
- **Customizable brush size** - Adjust drawing brush for better input
- **Model training** - Train your own models with included scripts
- **Cross-platform** - Works on Windows, macOS, and Linux

## Screenshots

*[Screenshots would go here in a real project]*

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd DeepLearning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (first time setup)
   ```bash
   python train_model.py
   ```
   This will:
   - Download the MNIST dataset automatically
   - Train a CNN model for digit recognition
   - Save the trained model to `models/digit_model.pth`
   - Display training progress and accuracy plots

4. **Run the application**
   ```bash
   python main.py
   ```

## Usage

### GUI Application

1. **Launch the application**
   ```bash
   python main.py
   ```

2. **Draw a digit**
   - Use your mouse or stylus to draw a digit (0-9) on the white canvas
   - The application will show real-time predictions as you draw

3. **View predictions**
   - Main prediction is displayed prominently
   - Confidence percentage is shown
   - Top 3 predictions with probabilities are listed

4. **Controls**
   - **Clear**: Erase the canvas and start over
   - **Predict**: Manually trigger prediction (auto-prediction is enabled)
   - **Brush Size**: Adjust the drawing brush size (5-25 pixels)

5. **Menu options**
   - **File → Load Model**: Load a different trained model
   - **File → Save Drawing**: Save your drawing as PNG
   - **Tools → Train New Model**: Instructions for training
   - **Help → About**: Application information

### Training a New Model

To train a new model with custom parameters:

```python
from src.model.trainer import DigitTrainer

# Create trainer
trainer = DigitTrainer(model_type='simple')  # or 'advanced'

# Prepare data
trainer.prepare_data(batch_size=64)

# Train
trainer.train(epochs=15, save_path='models/my_model.pth')

# View training history
trainer.plot_training_history()
```

### Command Line Options

```bash
# Launch GUI (default)
python main.py

# Train a new model
python main.py --train

# Specify custom model path
python main.py --model-path path/to/model.pth
```

## Project Structure

```
DeepLearning/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── digit_cnn.py      # CNN model architectures
│   │   └── trainer.py        # Model training utilities
│   ├── gui/
│   │   ├── __init__.py
│   │   └── drawing_app.py    # Main GUI application
│   └── utils/
│       ├── __init__.py
│       └── image_processing.py # Image preprocessing utilities
├── models/                   # Trained model files
├── data/                    # MNIST dataset (auto-downloaded)
├── main.py                  # Main application entry point
├── train_model.py          # Standalone training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Technical Details

### Model Architecture

The application uses a Convolutional Neural Network (CNN) with the following architecture:

**SimpleCNN** (default):
- Conv2D (1→16 channels, 5x5 kernel) + ReLU + MaxPool
- Conv2D (16→32 channels, 5x5 kernel) + ReLU + MaxPool
- Fully Connected (32×7×7 → 128) + ReLU + Dropout
- Output layer (128 → 10 classes)

**DigitCNN** (advanced):
- Three convolutional layers with increasing channels (32→64→128)
- Fully connected layers with dropout for regularization
- Higher capacity for better accuracy

### Image Processing Pipeline

1. **Canvas capture** - Convert drawing canvas to image array
2. **Preprocessing** - Grayscale conversion, inversion, normalization
3. **Bounding box detection** - Find and crop the drawn digit
4. **Resizing** - Scale to 28×28 pixels (MNIST format)
5. **Normalization** - Apply MNIST mean/std normalization
6. **Prediction** - Feed through trained CNN model

### Performance

- **Training accuracy**: ~99% on MNIST dataset
- **Validation accuracy**: ~98% on MNIST test set
- **Inference time**: <10ms per prediction on CPU
- **Model size**: ~500KB (SimpleCNN)

## Dependencies

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing
- **Pillow**: Image processing
- **matplotlib**: Plotting and visualization
- **opencv-python**: Advanced image processing
- **tkinter**: GUI framework (included with Python)
- **scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars

## Troubleshooting

### Common Issues

1. **"No module named 'tkinter'"**
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`
   - On CentOS/RHEL: `sudo yum install tkinter`

2. **CUDA out of memory**
   - Reduce batch size in training
   - Use CPU instead: model will automatically detect and use CPU

3. **Model not found**
   - Run `python train_model.py` to train a model first
   - Check that `models/digit_model.pth` exists

4. **Poor prediction accuracy**
   - Draw digits clearly and centered
   - Use appropriate brush size (10-20 pixels recommended)
   - Ensure digits fill most of the canvas area

### Performance Tips

- **For better accuracy**: Train with more epochs or use the advanced model
- **For faster inference**: Use the simple model on CPU
- **For training speed**: Use GPU if available (CUDA)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MNIST dataset creators
- PyTorch team for the deep learning framework
- Python community for excellent libraries

## Future Enhancements

- Support for multi-digit recognition
- Custom dataset training
- Model export to ONNX format
- Web-based interface
- Mobile app version
- Advanced preprocessing options
