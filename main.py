import cv2
from face_mesh import process_image as process_image_face_mesh
from face_detection import process_image as process_image_face_detection
from facial_recognition import FacialRecognition
import mediapipe as mp

# Constants
VIDEO_SOURCE = 0


def main(app_module):
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    # Create instances of FaceMesh, FaceDetection, and FacialRecognition
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    face_detection = mp.solutions.face_detection.FaceDetection()
    facial_recognition = FacialRecognition()
    facial_recognition.load_known_faces()

    try:
        while True:
            success, image = cap.read()
            if not success or image is None:
                break

            # Pass the instances as the second argument to the process_image functions
            if app_module == 'face_mesh':
                process_image_face_mesh(image, face_mesh)
            elif app_module == 'face_detection':
                process_image_face_detection(image, face_detection)
            elif app_module == 'facial_recognition':
                facial_recognition.recognize_and_mark_attendance(image)

            cv2.imshow("Image", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if app_module == 'facial_recognition':
            facial_recognition.save_attendance()


if __name__ == "__main__":
    print("Choose a module to run:")
    print("1. face_mesh")
    print("2. face_detection")
    print("3. facial_recognition")
    module_number = input("Enter the number of the module you want to run: ")

    module_dict = {
        '1': 'face_mesh',
        '2': 'face_detection',
        '3': 'facial_recognition'
    }

    module = module_dict.get(module_number, 'facial_recognition')
    main(module)
