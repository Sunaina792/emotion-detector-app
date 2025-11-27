# 🧠 Advanced Emotion Recognition System

A real-time emotion detection application powered by a custom-trained Convolutional Neural Network (CNN) that analyzes facial expressions and provides live analytics with an advanced, cyberpunk-inspired UI.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)

## 🎯 Features

### Core Functionality
- **Real-Time Emotion Detection**: Detects different emotions from live webcam feed
- **Custom CNN Model**: Trained on the FER2013 dataset with 30,000+ images
- **Multi-Face Detection**: Can detect and analyze multiple faces simultaneously
- **High Accuracy**: ~62% validation accuracy on complex emotion recognition tasks

### Advanced Analytics
- **Live Probability Distribution**: Real-time bar chart showing confidence levels for all 7 emotions
- **Session Statistics**: Histogram tracking emotion frequency during your session
- **Emotion History**: Tracks up to 100 recent emotion detections
- **Confidence Metrics**: Displays prediction confidence percentages

### User Interface
- **Modern Dark Theme**: Cyberpunk-inspired UI with neon blue accents
- **Responsive Layout**: Two-column design with live feed and analytics
- **Interactive Controls**: Toggle camera, probability charts, and history tracking
- **Real-Time Updates**: Smooth, lag-free video processing

## 🎭 Detected Emotions

The system can recognize the following emotions:

| Emotion | Emoji | Description |
|---------|-------|-------------|
| **Angry** | 😠 | Frustration, anger, or irritation |
| **Disgust** | 🤢 | Revulsion or strong disapproval |
| **Fear** | 😨 | Anxiety, worry, or fear |
| **Happy** | 😄 | Joy, happiness, or contentment |
| **Sad** | 😢 | Sadness, sorrow, or melancholy |
| **Surprise** | 😲 | Shock, amazement, or surprise |
| **Neutral** | 😐 | Calm, neutral expression |

## 🏗️ Project Structure

```
Emotion Detector/
├── app_advanced.py              # Advanced Streamlit application
├── main.py                      # OpenCV-based CLI version with CSV logging
├── emotion_detection_model.h5   # Trained CNN model weights
├── Emotion_Recognition_Train.ipynb  # Model training notebook
├── emotion_log.csv              # Session emotion logs (generated)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Webcam/Camera access
- 4GB+ RAM recommended

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/emotion-detector.git
   cd emotion-detector
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Option 1: Advanced Web Application (Recommended)

Run the feature-rich Streamlit application with analytics:

```bash
python -m streamlit run app_advanced.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Live probability distribution charts
- Session emotion statistics
- Modern, responsive UI
- Real-time analytics dashboard


### Option 2: Command-Line Version

Run the OpenCV-based version with CSV logging:

```bash
python main.py
```

**Features:**
- Native OpenCV window
- Automatic CSV logging to `emotion_log.csv`
- Press 'q' to quit

## 🧪 Model Architecture

The emotion detection model is a custom CNN with the following architecture:

![CNN MODEL architecture](cnn_architecture.png)

```
Input: 48x48 grayscale images
├── Conv2D (64 filters, 3x3) + ReLU
├── Conv2D (64 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Dropout (0.2)
├── Conv2D (128 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (128 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Dropout (0.22)
├── Flatten
├── Dense (512) + ReLU + Dropout (0.5)
├── Dense (256) + ReLU + Dropout (0.5)
└── Dense (7) + Softmax
```

**Training Details:**
- **Dataset**: FER2013 (30,000 training images, 2,300 validation images)
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 40
- **Batch Size**: 64
- **Data Augmentation**: Rotation, width/height shifts

**Performance:**
- Training Accuracy: ~73%
- Validation Accuracy: ~62.6%

## 📊 Dataset

The model was trained on the **FER2013** dataset:
- **Total Images**: 35,887
- **Image Size**: 48x48 pixels (grayscale)
- **Classes**: 7 emotions
- **Source**: Kaggle FER2013 Challenge

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **TensorFlow/Keras** | Deep learning framework for CNN model |
| **OpenCV** | Face detection and image processing |
| **Streamlit** | Web application framework |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation and CSV logging |
| **Altair** | Interactive data visualizations |
| **Pillow** | Image processing utilities |

## 📈 Performance Optimization

The application includes several optimizations:
- **Model Caching**: Uses `@st.cache_resource` to load model once
- **Efficient Face Detection**: Haar Cascade for fast real-time detection
- **Batch Prediction**: Processes multiple faces efficiently
- **Minimal Latency**: Optimized frame processing pipeline

## 🎨 UI Customization

The advanced app features a custom dark theme with:
- Neon blue accent colors (#00d4ff)
- Dark background (#0e1117)
- Styled metrics and charts
- Responsive two-column layout
- Custom CSS for modern aesthetics

## 📝 Future Enhancements

Potential improvements for future versions:
- [ ] Multi-language support
- [ ] Emotion trend analysis over time
- [ ] Export session reports as PDF
- [ ] Audio alerts for specific emotions
- [ ] Snapshot/screenshot functionality
- [ ] Cloud deployment (Streamlit Cloud/Heroku)
- [ ] Mobile app version
- [ ] Integration with video files (not just webcam)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FER2013 Dataset**: Kaggle Facial Expression Recognition Challenge
- **TensorFlow Team**: For the excellent deep learning framework
- **Streamlit Team**: For the intuitive web app framework
- **OpenCV Community**: For robust computer vision tools

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ and Python**
