import cv2
import matplotlib.pyplot as plt
import mediapipe as mp


class FaceLandmarkDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    @staticmethod
    def convert_image_to_rgb(image):
        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as error:
            print(f"An error occurred while converting the image to RGB: {error}")
            return None

    @staticmethod
    def process_image_with_processor(image, processor):
        try:
            return processor.process(image)
        except Exception as error:
            print(f"An error occurred while processing the image: {error}")
            return None

    @staticmethod
    def plot_face_blendshapes_bar_graph(face_blendshapes):
        blendshape_names = [blendshape.name for blendshape in face_blendshapes]
        blendshape_values = [blendshape.value for blendshape in face_blendshapes]

        plt.figure(figsize=(10, 5))
        plt.bar(blendshape_names, blendshape_values)
        plt.xlabel("Blendshape Name")
        plt.ylabel("Blendshape Value")
        plt.title("Face Blendshapes")
        plt.xticks(rotation=45)
        plt.show()

    def detect_face_landmarks(self, image):
        image_rgb = self.convert_image_to_rgb(image)
        if image_rgb is None:
            return

        results = self.process_image_with_processor(image_rgb, self.face_mesh)
        if results is None:
            return

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                annotated_image = self.draw_landmarks_on_image(image, face_landmarks)
                cv2.imshow("Annotated Image", annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                self.plot_face_landmarks(face_landmarks)

    def detect_face_blendshapes(self, image):
        image_rgb = self.convert_image_to_rgb(image)
        if image_rgb is None:
            return

        results = self.process_image_with_processor(image_rgb, self.face_mesh)
        if results is None:
            return

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_blendshapes = face_landmarks.face_blend_shapes
                self.plot_face_blendshapes_bar_graph(face_blendshapes)

    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        image = rgb_image.copy()
        for landmark in detection_result.landmark:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        return image

    @staticmethod
    def plot_face_landmarks(face_landmarks):
        x_coordinates = [landmark.x for landmark in face_landmarks.landmark]
        y_coordinates = [landmark.y for landmark in face_landmarks.landmark]
        z_coordinates = [landmark.z for landmark in face_landmarks.landmark]

        plt.figure(figsize=(10, 5))
        plt.plot(x_coordinates, label='X')
        plt.plot(y_coordinates, label='Y')
        plt.plot(z_coordinates, label='Z')
        plt.xlabel("Landmark index")
        plt.ylabel("Coordinate value")
        plt.title("Face Landmarks")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    video_source = 0  # Use default camera
    cap = cv2.VideoCapture(video_source)

    detector = FaceLandmarkDetector()

    while True:
        success, frame = cap.read()
        if not success:
            break

        detector.detect_face_landmarks(frame)
        detector.detect_face_blendshapes(frame)

    cap.release()
    cv2.destroyAllWindows()
