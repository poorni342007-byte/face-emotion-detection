# Face and Emotion Detection using OpenCV & DeepFace

This mini-project performs real-time face detection and emotion recognition
using OpenCV and the DeepFace library.

## Features
- Real-time face detection using Haar Cascade
- Emotion recognition (happy, sad, angry, neutral, etc.)
- Stable emotion output using temporal smoothing
- Face bounding box with emotion label

## Technologies Used
- Python
- OpenCV
- DeepFace
- TensorFlow

## Working Principle
1. Webcam frames are captured using OpenCV
2. Faces are detected using Haar Cascade classifier
3. Emotion detection is applied on detected face regions
4. Emotion output is stabilized using majority voting over frames

## Known Limitation
- Real-time emotion detection using DeepFace may be unstable on Python 3.13.x
  due to TensorFlow compatibility issues.
- This is an environment-level limitation, not a code issue.

## Future Improvements
- Use a lightweight CNN model
- Improve real-time performance
- Add support for multiple faces