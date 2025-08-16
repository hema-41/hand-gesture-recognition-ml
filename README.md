# Hand Gesture Recognition – Machine Learning

This project implements a **real-time hand gesture recognition system** using **MediaPipe** and **OpenCV**, with evaluation metrics for accuracy, precision, recall, and F1-score.  

## 🚀 Features
- Real-time hand tracking and gesture recognition.  
- Utilizes **MediaPipe Hands** for landmark detection.  
- Draws hand landmarks and connections on live video.  
- Evaluates predictions with metrics: Accuracy, Precision, Recall, F1-Score.  
- Calculates average frame processing time.  
- Includes comparison with **state-of-the-art benchmarks** (placeholders).  

## 🛠️ Tech Stack
- Python  
- OpenCV  
- MediaPipe  
- scikit-learn  

## 📌 Methodology
1. **Hand Detection & Tracking**  
   - MediaPipe’s `Hands` module detects 21 hand landmarks in real-time.  
   - Landmarks are drawn on each frame using OpenCV.  

2. **Gesture Classification**  
   - Placeholder functions (`get_true_label()` & `get_predicted_label()`) simulate label assignment.  
   - Can be extended to classify multiple hand gestures.  

3. **Performance Metrics**  
   - Accuracy, Precision, Recall, F1-Score calculated dynamically.  
   - Average frame processing time displayed.  

4. **Comparison with State-of-the-Art**  
   - Placeholder metrics included for benchmarking.  

## 🔧 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition-ml.git
