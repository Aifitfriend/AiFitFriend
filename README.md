# Description
## AI FitFriend : Your AI Workout Companion 
Maintaining a state of physical and health well-being is a challenging task, especially when exercises are performed at home without proper guidance. The absence of instructors and the lack of proper equipment can lead to incorrect execution of postures and exercises. As a way to resolve this problem, our team has come up with a solution to address improper posture during workouts. Users can select their preferred fitness program and engage in it as daily exercises. In the event that our model detects any incorrect posture, it will automatically pause the exercise and display a video demonstrating the correct form. Through consistent use of this application, users can efficiently achieve and maintain their personal fitness goals, even in the comfort of their homes!

# Workout Pose Detection, Counter, and Pose Correction
The problem with the ordinary workout application found in the market is the absence of a feedback mechanism that ensures the correctness of movements.

The algorithm consists of 2 machine learning models, which are human body keypoints detection and action recognition.

## Human Body Keypoints Detection
The human body keypoints detection model was built by fine-tuning the pre-trained CenterNet HourGlass104 Keypoints 512x512 model. This model is trained using a dataset containing the keypoint's label and coordinates values.

## Pose Recognition
The pose recognition model was built using LSTM-Attention architecture. This model is trained using a dataset containing the keypoints coordinates values and the action label.

## Counters and Pose Correction Feedback
The algorithm for counters and pose correction was made using the calculation logic of the angles formed by several key points.

# Dataset
Model 1:

Model 2:
The dataset containing workout videos can be seen [here](https://drive.google.com/drive/folders/1Nvg6nRuQ8j4N77hJIoFZn0xIsP1NpWKP?usp=sharing)

The human body keypoints were then extracted using mediapipe library and this keypoints sequence were then be used as the dataset for the second model. The keypoints sequence dataset can be seen [here](https://drive.google.com/drive/folders/1fArEPdnxRTAx3EzPEnXnTr2vcPcTT9dX?usp=sharing)

