# AMS Face Recognition

AMS Face Recognition is a Python application that uses computer vision techniques to recognize faces and mark attendance. It leverages MediaPipe and OpenCV for face detection, the `face_recognition` library for face recognition, and includes features for visualizing face landmarks and blendshapes.

## Features

- **Face Detection**: Utilizes MediaPipe and OpenCV to detect faces in real-time.
- **Face Recognition**: Uses the `face_recognition` library to identify known faces.
- **Attendance Marking**: Automatically marks attendance for recognized faces.
- **Visualization**: Displays face landmarks and blendshapes for analysis.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/bantoinese83/AMS-Face-Recognition.git
    cd AMS-Face-Recognition
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application:**
    ```bash
    python main.py
    ```

## Usage

1. **Prepare Known Faces:**
    - Store images of known faces in a directory named `students`. Each image file name should correspond to the name of the person (e.g., `john_doe.jpg`).

2. **Run the Application:**
    - Upon running the application, you'll be prompted to choose a module to run:
        - `1. face_mesh`
        - `2. face_detection`
        - `3. facial_recognition`
    - Enter the number corresponding to the module you want to execute.

3. **Modules:**
    - **Face Mesh**: Visualizes face landmarks.
    - **Face Detection**: Detects faces and draws bounding boxes.
    - **Facial Recognition**: Recognizes faces and marks attendance.

4. **Exit the Application:**
    - Press `q` to quit the application.

## Example

To run the facial recognition module, follow these steps:

1. **Prepare the environment:**
    - Place images of known individuals in the `students` directory.

2. **Execute the script:**
    ```bash
    python main.py
    ```

3. **Select the module:**
    - Enter `3` to run the facial recognition module.

## Logging and Debugging

- **Logging**: The application uses `loguru` for logging. Logs are printed to the console for real-time debugging.
- **Error Handling**: The code includes try-except blocks to handle potential errors during image processing and logging relevant messages.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, feel free to open an issue or submit a pull request.

1. **Fork the repository.**
2. **Create a new branch:**
    ```bash
    git checkout -b feature-branch
    ```
3. **Commit your changes:**
    ```bash
    git commit -m "Add some feature"
    ```
4. **Push to the branch:**
    ```bash
    git push origin feature-branch
    ```
5. **Open a pull request.**

## Acknowledgements

- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [loguru](https://github.com/Delgan/loguru)
- [rich](https://github.com/Textualize/rich)

## Contact

For any inquiries, please contact `b.antoine.se@gmail.com`.
