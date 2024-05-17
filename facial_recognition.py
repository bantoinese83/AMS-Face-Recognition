import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console

console = Console()


class FacialRecognition:
    def __init__(self, students_dir='students'):
        self.known_face_encodings = []
        self.known_face_names = []
        self.students_dir = students_dir
        self.attendance = pd.DataFrame(columns=["Name", "Date", "Time"])

    def load_known_faces(self):
        for name in os.listdir(self.students_dir):
            image = face_recognition.load_image_file(f"{self.students_dir}/{name}")
            encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
        logger.info("Loaded known faces")

    def recognize_and_mark_attendance(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            self.mark_attendance(name)
        self.draw_attendees_name_on_frame(frame)

    def mark_attendance(self, name):
        now = datetime.now()
        date = now.date()
        time = now.time()
        new_record = pd.DataFrame({"Name": [name], "Date": [date], "Time": [time]})
        self.attendance = pd.concat([self.attendance, new_record], ignore_index=True)
        logger.info(f"Marked attendance for {name} at {time} on {date}")
        console.print(f"Marked attendance for {name} at {time} on {date}", style="bold green")

    def save_attendance(self, filename="attendance.csv"):
        self.attendance.to_csv(filename, index=False)
        logger.info(f"Saved attendance to {filename}")
        console.print(f"Saved attendance to {filename}", style="bold blue")

    def show_attendance(self):
        console.print(self.attendance, style="bold magenta")
        logger.info("Displayed attendance")

    def draw_attendees_name_on_frame(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
