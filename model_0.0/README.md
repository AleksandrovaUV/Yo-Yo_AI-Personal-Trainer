**In this folder you can find a functional OpenPose exploration model.**  
The model uses existent frameworks and can process both static images and video streams to estimate human body poses. It leverages OpenCV’s `dnn` module to load a pre-trained TensorFlow model (`graph_opt.pb`) and detect key body parts based on confidence heatmaps.

### Key Features:
- **Pose Estimation**: The script identifies 18 key body parts (e.g., nose, shoulders, elbows, knees) and connects them using predefined pairs to visualize the human skeleton.
- **Image & Video Support**: It supports both single-image inference and real-time pose estimation from video files or webcam input.
- **Visualization**: Detected poses are drawn directly on the input frame using colored lines and circles, and can be displayed using either OpenCV windows or matplotlib (depending on the environment).
- **Modular Design**: The core logic is encapsulated in a `pose_estimation()` function, making it easy to reuse or extend.

### Technologies Used:
- **OpenCV (`cv2`)**: For image processing, neural network inference, and visualization.
- **Matplotlib**: For displaying results in environments without GUI support.
- **TensorFlow Model**: The model file `graph_opt.pb` contains the trained weights for pose detection.
