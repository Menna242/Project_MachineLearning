import cv2
import numpy as np
import joblib
import time
from skimage.feature import hog
class RealtimeApp:
    def __init__(self, model_path="svm_model_both.pkl"):
        print("[INFO] Loading SVM model...")

        model_data = joblib.load(model_path)

        self.model = model_data['svm_model']
        self.scaler = model_data['scaler']
        self.feature_type = 'both'
        self.confidence_threshold = model_data['confidence_threshold']

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

        print("[INFO] Model loaded successfully")

    # ---------- Feature Extraction ----------
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

    def extract_features(self, image):
        color_feat = self.get_color_features(image)
        hog_feat = self.get_hog_features(image)
        return np.concatenate([color_feat, hog_feat])

    # ---------- Prediction ----------
    def predict(self, frame):
        features = self.extract_features(frame).reshape(1, -1)
        features = self.scaler.transform(features)

        probs = self.model.predict_proba(features)[0]
        confidence = np.max(probs)
        pred_class = np.argmax(probs)

        if confidence < self.confidence_threshold:
            pred_class = 6  # Unknown

        return pred_class, confidence

    # ---------- Drawing ----------
    def draw_overlay(self, frame, class_id, confidence):
        h, w = frame.shape[:2]
        label = self.class_names[class_id]
        color = self.class_colors[class_id]

        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    color, 3)

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

    # ---------- Main Loop ----------
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

            class_id, confidence = self.predict(frame)
            frame = self.draw_overlay(frame, class_id, confidence)
            self.update_fps()

            cv2.imshow("Real-time Material Classification (SVM)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Application closed")


# ---------- Run ----------
if __name__ == "__main__":
    app = RealtimeApp("svm_model_both.pkl")
    app.run()
