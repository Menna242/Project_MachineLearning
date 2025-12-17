import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def apply_minimum_rejection(avg_distances, rejected, min_rate=0.05):
    n = len(avg_distances)
    target = int(min_rate * n)

    if rejected.sum() >= target:
        return rejected

    # reject the largest distances
    idx = np.argsort(avg_distances)[-target:]
    rejected[idx] = True
    return rejected

class KNNClassifierWithRejection:
    def __init__(self, k=3, rejection_threshold=None, n_components=None, feature_type='color'):
        self.k = k
        self.rejection_threshold = rejection_threshold
        self.feature_type = feature_type

        # PCA Auto-tuning
        if n_components is None:
            if feature_type == 'color':
                self.n_components = 100
            elif feature_type == 'hog':
                self.n_components = 150
            else:
                self.n_components = 200
        else:
            self.n_components = n_components

        metric = 'euclidean'


        self.model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, whiten=False)

        self.class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]
        self.n_primary_classes = 6
        self.unknown_class_id = 6

    def train(self, X_train, y_train):
        primary_mask = y_train < self.n_primary_classes
        X_train_p = X_train[primary_mask]
        y_train_p = y_train[primary_mask]

        X_scaled = self.scaler.fit_transform(X_train_p)
        actual_n = min(self.n_components, X_scaled.shape[0], X_scaled.shape[1])
        if actual_n != self.pca.n_components:
            self.pca = PCA(n_components=actual_n, whiten=True)

        X_pca = self.pca.fit_transform(X_scaled)
        self.model.fit(X_pca, y_train_p)

    def _preprocess(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def calculate_threshold(self, X_train, y_train):
        primary_mask = y_train < self.n_primary_classes
        X_train_primary = X_train[primary_mask]

        X_processed = self._preprocess(X_train_primary)
        distances, _ = self.model.kneighbors(X_processed)
        avg_distances = distances.mean(axis=1)

        # Logic based on feature_type
        if self.feature_type == 'color':
            percentile = 99
            threshold = np.percentile(avg_distances, percentile)
            print(f"\n📊 Rejection Threshold (Percentile-based for Color):")
            print(f"   Percentile: {percentile}th")
        elif self.feature_type == 'hog':
            std_factor = 4.0
            mean = avg_distances.mean()
            std = avg_distances.std()
            threshold = mean + std_factor * std
            print(f"\n📊 Rejection Threshold (STD-based for HOG):")
            print(f"   Mean: {mean:.4f}")
            print(f"   Std : {std:.4f}")
        else:  # Combined
            percentile = 99.5
            threshold = np.percentile(avg_distances, percentile)
            print(f"\n📊 Rejection Threshold (Balanced Percentile for Combined):")
            print(f"   Percentile: {percentile}th")

        print(f"   Threshold: {threshold:.4f}")
        return threshold

    def predict_with_rejection(self, X_test):
        X_processed = self._preprocess(X_test)
        distances, _ = self.model.kneighbors(X_processed)
        avg_distances = distances.mean(axis=1)

        base_predictions = self.model.predict(X_processed)

        rejected = avg_distances > self.rejection_threshold

        final_predictions = base_predictions.copy()
        final_predictions[rejected] = self.unknown_class_id
        return final_predictions, rejected, avg_distances

    def evaluate(self, X_test, y_test, show_details=True):
        predictions, rejected, distances = self.predict_with_rejection(X_test)
        primary_mask = y_test < self.n_primary_classes

        valid = primary_mask & (predictions != self.unknown_class_id)

        accuracy_primary = (
            accuracy_score(y_test[valid], predictions[valid])
            if valid.sum() > 0 else 0
        )

        rejection_rate = rejected.sum() / len(rejected) * 100

        false_rejections = primary_mask & rejected
        false_rejection_rate = (false_rejections.sum() / primary_mask.sum() * 100) if primary_mask.sum() > 0 else 0

        overall_accuracy = accuracy_score(y_test, predictions)

        print(f"\n PRIMARY CLASSES PERFORMANCE:")
        print(f"   Achieved Accuracy: {accuracy_primary*100:.2f}%")
        print(f"    False Rejection Rate: {false_rejection_rate:.2f}%")
        print(f"   Total Rejection Rate: {rejection_rate:.2f}%")

        if show_details and primary_mask.sum() > 0:
            print(f"\n CLASSIFICATION REPORT (Primary):")
            y_true = y_test[primary_mask]
            y_pred = np.clip(predictions[primary_mask], 0, 5)
            print(classification_report(y_true, y_pred, target_names=self.class_names[:6], zero_division=0))

        cm = confusion_matrix(y_test[primary_mask], np.clip(predictions[primary_mask], 0, 5), labels=range(6)) if primary_mask.sum() > 0 else np.zeros((6,6))

        return {
            'accuracy_primary': accuracy_primary,
            'rejection_rate': rejection_rate,
            'false_rejection_rate': false_rejection_rate,
            'confusion_matrix': cm,
            'meets_requirement': accuracy_primary >= 0.85
        }

    def save_model(self, filename):
        model_data = {
            'model': self.model, 'scaler': self.scaler, 'pca': self.pca,
            'k': self.k, 'rejection_threshold': self.rejection_threshold,
            'feature_type': self.feature_type, 'class_names': self.class_names
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        knn = KNNClassifierWithRejection(
            k=data['k'],
            feature_type=data['feature_type']
        )
        knn.model = data['model']
        knn.scaler = data['scaler']
        knn.pca = data['pca']
        knn.rejection_threshold = data['rejection_threshold']
        knn.class_names = data['class_names']
        return knn


class CombinedKNNWithRejection:
    def __init__(self, k=3, w_color=0.65, w_hog=0.35):
        self.k = k
        self.wc = w_color
        self.wh = w_hog

        self.color_knn = KNNClassifierWithRejection(k=k, feature_type='color')
        self.hog_knn   = KNNClassifierWithRejection(k=k, feature_type='hog')

        self.threshold = None
        self.unknown_class_id = 6

    def train(self, X_color, X_hog, y):
        self.color_knn.train(X_color, y)
        self.hog_knn.train(X_hog, y)

    def calculate_threshold(self, X_color, X_hog, y):
        mask = y < 6

        Dc, _ = self.color_knn.model.kneighbors(
            self.color_knn._preprocess(X_color[mask])
        )
        Dh, _ = self.hog_knn.model.kneighbors(
            self.hog_knn._preprocess(X_hog[mask])
        )

        fused_dist = self.wc * Dc.mean(axis=1) + self.wh * Dh.mean(axis=1)

        # Same percentile logic you already use
        self.threshold = np.percentile(fused_dist, 99.7)

        print("\n Rejection Threshold (Combined – Distance Fusion):")
        print(f"   Threshold: {self.threshold:.4f}")

    def predict_with_rejection(self, X_color, X_hog):
        Dc, _ = self.color_knn.model.kneighbors(
            self.color_knn._preprocess(X_color)
        )
        Dh, _ = self.hog_knn.model.kneighbors(
            self.hog_knn._preprocess(X_hog)
        )

        fused_dist = self.wc * Dc.mean(axis=1) + self.wh * Dh.mean(axis=1)

        preds = self.color_knn.model.predict(
            self.color_knn._preprocess(X_color)
        )

        rejected = fused_dist > self.threshold
        preds[rejected] = self.unknown_class_id

        return preds, rejected


def compare_k_values(X_train, X_test, y_train, y_test, feature_name, feature_type, k_values=[3, 5], pca_components=None):
    results = {}
    print(f"\n{'#'*70}\nTESTING: {feature_name}\n{'#'*70}")

    for k in k_values:
        knn = KNNClassifierWithRejection(k=k, n_components=pca_components, feature_type=feature_type)
        knn.train(X_train, y_train)

        # New call style
        knn.rejection_threshold = knn.calculate_threshold(X_train, y_train)

        result = knn.evaluate(X_test, y_test, show_details=(k==3))
        results[k] = {'model': knn, 'metrics': result}

    print(f"\nSUMMARY - {feature_name}")
    print(f"{'k':<5} {'Accuracy':<12} {'Rejection':<12} {'False Rej':<12}")
    print("-" * 50)
    for k in k_values:
        m = results[k]['metrics']
        print(f"{k:<5} {m['accuracy_primary']*100:<12.2f} "
              f"{m['rejection_rate']:<12.2f} "
              f"{m['false_rejection_rate']:<12.2f}")
    return results


def main():
    features_color = np.load("features_color.npy")
    features_hog   = np.load("features_hog.npy")
    labels         = np.load("labels.npy")


    Xc_train, Xc_test, Xh_train, Xh_test, y_train, y_test = train_test_split(
        features_color,
        features_hog,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    feature_types = [("Color Hist", "color", Xc_train, Xc_test),
                     ("HOG Features", "hog", Xh_train, Xh_test)]

    k_values = [3, 5]
    primary_mask = y_test < 6

    for feature_name, feature_type, X_train, X_test in feature_types:
        print(f"\n{'#'*70}\nTESTING: {feature_name}\n{'#'*70}")

        results = {}
        for k in k_values:
            knn = KNNClassifierWithRejection(k=k, feature_type=feature_type)
            knn.train(X_train, y_train)
            knn.rejection_threshold = knn.calculate_threshold(X_train, y_train)


            knn.save_model(f"knn_{feature_type}_k{k}.pkl")
            print(f"✓ Saved {feature_type} KNN model with k={k}")
            preds, rejected, _ = knn.predict_with_rejection(X_test)

            valid = primary_mask & (preds != 6)
            acc = accuracy_score(y_test[valid], preds[valid])
            false_rej = (rejected & primary_mask).sum() / primary_mask.sum() * 100
            total_rej = rejected.mean() * 100

            results[k] = (acc, false_rej, total_rej)

            print(f"\n k = {k}")
            print(f" Accuracy: {acc*100:.2f}%")
            print(f" False Rejection Rate: {false_rej:.2f}%")
            print(f" Total Rejection Rate: {total_rej:.2f}%")

        print("\n Comparison Table:")
        print(f"{'k':<5} {'Accuracy (%)':<15} {'False Rejection (%)':<20} {'Total Rejection (%)':<20}")
        print("-"*65)
        for k in k_values:
            acc, fr, tr = results[k]
            print(f"{k:<5} {acc*100:<15.2f} {fr:<20.2f} {tr:<20.2f}")


    print(f"\n{'#'*70}\nTESTING: Combined (Distance Fusion)\n{'#'*70}")
    combined_results = {}
    for k in k_values:
        combined = CombinedKNNWithRejection(k=k)
        combined.train(Xc_train, Xh_train, y_train)
        combined.calculate_threshold(Xc_train, Xh_train, y_train)

        combined.color_knn.rejection_threshold = combined.color_knn.calculate_threshold(Xc_train, y_train)
        combined.hog_knn.rejection_threshold = combined.hog_knn.calculate_threshold(Xh_train, y_train)

        combined.color_knn.save_model(f"combined_knn_k{k}_color.pkl")
        combined.hog_knn.save_model(f"combined_knn_k{k}_hog.pkl")


        preds, rejected = combined.predict_with_rejection(Xc_test, Xh_test)

        valid = primary_mask & (preds != 6)
        acc = accuracy_score(y_test[valid], preds[valid])
        false_rej = (rejected & primary_mask).sum() / primary_mask.sum() * 100
        total_rej = rejected.mean() * 100

        combined_results[k] = (acc, false_rej, total_rej)

        print(f"\n k = {k}")
        print(f" COMBINED Accuracy: {acc*100:.2f}%")
        print(f" False Rejection Rate: {false_rej:.2f}%")
        print(f" Total Rejection Rate: {total_rej:.2f}%")

    print("\n Combined Comparison Table:")
    print(f"{'k':<5} {'Accuracy (%)':<15} {'False Rejection (%)':<20} {'Total Rejection (%)':<20}")
    print("-"*65)
    for k in k_values:
        acc, fr, tr = combined_results[k]
        print(f"{k:<5} {acc*100:<15.2f} {fr:<20.2f} {tr:<20.2f}")

if __name__ == "__main__":
    main()
