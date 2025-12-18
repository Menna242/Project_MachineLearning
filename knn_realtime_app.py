import cv2
import numpy as np
import time
from skimage.feature import hog
from knn_train import KNNClassifierWithRejection, CombinedKNNWithRejection


class RealtimeApp:
    def __init__(self, model_prefix="combined_knn_k3"):
        print("[INFO] Loading Combined KNN models (k=3)...")

        self.knn = CombinedKNNWithRejection(k=3)
        self.knn.color_knn = KNNClassifierWithRejection.load_model(
            model_prefix + "_color.pkl"
        )
        self.knn.hog_knn = KNNClassifierWithRejection.load_model(
            model_prefix + "_hog.pkl"
        )

        self.knn.threshold = (
            0.65 * self.knn.color_knn.rejection_threshold +
            0.35 * self.knn.hog_knn.rejection_threshold
        )

        print("[INFO] Combined KNN loaded successfully")

        self.class_names = {
            0: 'Glass',
            1: 'Paper',
            2: 'Cardboard',
            3: 'Plastic',
            4: 'Metal',
            5: 'Trash',
            6: 'Unknown'
        }

        self.class_colors = {
            0: (255, 200, 100),
            1: (200, 200, 200),
            2: (100, 150, 200),
            3: (0, 255, 255),
            4: (180, 180, 180),
            5: (0, 0, 200),
            6: (100, 100, 100)
        }

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def get_color_features(self, image):
        image = cv2.resize(image, (128, 128))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv], [0, 1, 2],
            None, [8, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )

        cv2.normalize(hist, hist)
        return hist.flatten()

    def get_hog_features(self, image):
        image = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

    def predict(self, frame):
        X_color = self.get_color_features(frame).reshape(1, -1)
        X_hog   = self.get_hog_features(frame).reshape(1, -1)

        preds, rejected = self.knn.predict_with_rejection(X_color, X_hog)

        if rejected[0] or preds[0] == 6:
            return 6  # Unknown

        return preds[0]


    def draw_overlay(self, frame, class_id):
        h, w = frame.shape[:2]
        label = self.class_names[class_id]
        color = self.class_colors[class_id]

        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

        text = f"{label}"
        cv2.putText(frame, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5,color, 3)

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        return frame

    def update_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time

        if elapsed >= 1:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return

        print("[INFO] Camera started")
        print("Press 'q' to exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            class_id = self.predict(frame)
            frame = self.draw_overlay(frame, class_id)
            self.update_fps()

            cv2.imshow("Real-time Material Classification (KNN Combined)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Application closed")

if __name__ == "__main__":
    app = RealtimeApp("combined_knn_k3")
    app.run()
