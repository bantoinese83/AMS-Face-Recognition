# face mesh

import time

import cv2
import mediapipe as mp
from loguru import logger

# Constants
VIDEO_SOURCE = 0  # Use default camera
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 3
FONT_COLOR = (255, 0, 0)
FONT_THICKNESS = 2


def draw_landmarks(image, face_landmarks):
    draw_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2)
    mp.solutions.drawing_utils.draw_landmarks(image, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                              draw_spec, draw_spec)

    # Calculate the bounding box of the face
    landmark_list = [landmark for landmark in face_landmarks.landmark]
    x_coordinates = [landmark.x for landmark in landmark_list]
    y_coordinates = [landmark.y for landmark in landmark_list]
    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)

    # Draw a circle around the face
    x_center = int((x_min + x_max) / 2 * image.shape[1])
    y_center = int((y_min + y_max) / 2 * image.shape[0])
    radius = max(int((x_max - x_min) / 2 * image.shape[1]), int((y_max - y_min) / 2 * image.shape[0]))
    cv2.circle(image, (x_center, y_center), radius, (0, 255, 0), 2)


def process_image(image, face_mesh):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as error:
        logger.error(f"An error occurred while converting the image to RGB: {error}")
        return

    try:
        results = face_mesh.process(image_rgb)
    except Exception as error:
        logger.error(f"An error occurred while processing the image: {error}")
        return

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            draw_landmarks(image, face_landmarks)


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    previous_time = 0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        min_detection_confidence=0.3,  # Lower this value to make detection more sensitive
        min_tracking_confidence=0.3  # Lower this value to make tracking more sensitive
    )

    try:
        while True:
            success, image = cap.read()
            if not success or image is None:
                break

            process_image(image, face_mesh)

            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            cv2.putText(image, f'FPS: {int(fps)}', (20, 70), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            cv2.imshow("Image", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
