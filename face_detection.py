import time

import cv2
import mediapipe as mp
from loguru import logger

# Constants
VIDEO_SOURCE = 0  # Use default camera
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 0, 0)
FONT_THICKNESS = 2
BOX_COLOR = (255, 0, 255)
BOX_THICKNESS = 2


def draw_detection(image, detection):
    image_height, image_width, image_channels = image.shape
    bounding_box_coordinates = detection.location_data.relative_bounding_box
    bounding_box = (int(bounding_box_coordinates.xmin * image_width),
                    int(bounding_box_coordinates.ymin * image_height),
                    int(bounding_box_coordinates.width * image_width),
                    int(bounding_box_coordinates.height * image_height))
    cv2.rectangle(image, bounding_box, BOX_COLOR, BOX_THICKNESS)
    cv2.putText(image, f'{int(detection.score[0] * 100)}%',
                (bounding_box[0], bounding_box[1] - 20), FONT, FONT_SCALE, BOX_COLOR, FONT_THICKNESS)


def log_detection_details(detection_id, detection):
    logger.info(f"Detection {detection_id}: {detection}")
    logger.info(f"Location data: {detection.location_data}")
    logger.info(f"Relative bounding box: {detection.location_data.relative_bounding_box}")
    logger.info(f"Score: {detection.score[0]}")


def process_image(image, face_detection):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as error:
        logger.error(f"An error occurred while converting the image to RGB: {error}")
        return

    try:
        results = face_detection.process(image_rgb)
    except Exception as error:
        logger.error(f"An error occurred while processing the image: {error}")
        return

    mp_draw = mp.solutions.drawing_utils

    if results and results.detections:
        for detection_id, detection in enumerate(results.detections):
            try:
                mp_draw.draw_detection(image, detection)
                log_detection_details(detection_id, detection)
                draw_detection(image, detection)
            except Exception as error:
                logger.error(f"An error occurred while drawing the detection: {error}")
                continue

            bbox = detection.location_data.relative_bounding_box
            image_height, image_width, _ = image.shape
            bbox.xmin = int(bbox.xmin * image_width)
            bbox.ymin = int(bbox.ymin * image_height)
            bbox.width = int(bbox.width * image_width)
            bbox.height = int(bbox.height * image_height)
            logger.info(f"Bounding box: {bbox}")

            # Log landmarks
            for landmark in detection.location_data.relative_keypoints:
                logger.info(f"Landmark: {landmark}")
    else:
        logger.info("No detections found")


def main():
    previous_time = 0
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    try:
        while True:
            success, image = cap.read()
            if not success or image is None:
                break

            process_image(image, face_detection)

            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            cv2.putText(image, f"FPS: {int(fps)}", (20, 50), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            cv2.imshow("Image", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

