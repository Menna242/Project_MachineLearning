import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class SVMClassifier:
    def __init__(self, feature_type='both'):
        self.feature_type = feature_type
        self.scaler = StandardScaler()
        self.svm_model = None
        self.confidence_threshold = 0.5
        self.class_names = classes = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]

    def load_data(self):
        print(f"\n{'='*60}")
        print(f"{self.feature_type} features:")
        print(f"{'='*60}")
        
        if self.feature_type == 'color':
            self.features = np.load('features_color.npy')
        elif self.feature_type == 'hog':
            self.features = np.load('features_hog.npy')
        elif self.feature_type == 'both':
            self.features = np.load('features_both.npy')
        else:
            raise ValueError("feature_type must be 'color', 'hog', or 'both'")
        
        self.labels = np.load('labels.npy')
        
        print(f"Number of samples: {len(self.labels)}")
        print(f"Feature vector length: {self.features.shape[1]}")
        
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nClass distribution:")
        for cls, count in zip(unique, counts):
            print(f" {self.class_names[cls]}: {count} samples")


    def prepare_data(self, test_size=0.2, random_state=42):
        print(f"\n{'='*60}")
        print("Splitting data into train and validation sets:")
        print(f"{'='*60}")
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.features, 
            self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels  
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        
        # Standardize features 
        print("\nStandardizing features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)


    def train_simple(self, kernel='rbf', C=10, gamma='scale'):
        print(f"\n{'='*60}")
        print(f"Training SVM with {kernel} kernel...")
        print(f"{'='*60}")
        
        self.svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True, 
            class_weight='balanced',
            random_state=42
        )
        
        self.svm_model.fit(self.X_train_scaled, self.y_train)
        print("Training complete")

    def evaluate(self):
        print(f"\n{'='*60}")
        print("Evaluating Model Performance")
        print(f"{'='*60}")
        
        y_pred = self.svm_model.predict(self.X_val_scaled)
        
        accuracy = accuracy_score(self.y_val, y_pred)
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            self.y_val, 
            y_pred, 
            target_names=self.class_names,
            digits=4
        ))
        
        return accuracy
    
    def evaluate_with_rejection(self, confidence_threshold=0.5):
        print(f"\n{'='*60}")
        print("Evaluating with Unknown Class Rejection")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"{'='*60}")
        
        self.confidence_threshold = confidence_threshold
        
        y_proba = self.svm_model.predict_proba(self.X_val_scaled)
        max_proba = np.max(y_proba, axis=1)
        
        y_pred_with_rejection = []
        for i, proba in enumerate(max_proba):
            if proba < confidence_threshold:
                y_pred_with_rejection.append(6)  # Unknown class
            else:
                y_pred_with_rejection.append(self.svm_model.predict(self.X_val_scaled[i:i+1])[0])
        
        y_pred_with_rejection = np.array(y_pred_with_rejection)
        
        n_rejected = np.sum(y_pred_with_rejection == 6)
        rejection_rate = n_rejected / len(y_pred_with_rejection)
        
        print(f"\nRejection Statistics:")
        print(f"  Samples classified as Unknown: {n_rejected}")
        print(f"  Rejection rate: {rejection_rate:.2%}")
        
        non_rejected_mask = y_pred_with_rejection != 6
        if np.sum(non_rejected_mask) > 0:
            accuracy_non_rejected = accuracy_score(
                self.y_val[non_rejected_mask],
                y_pred_with_rejection[non_rejected_mask]
            )
            print(f"Accuracy on non-rejected samples: {accuracy_non_rejected:.4%}")
        
        return accuracy_non_rejected, rejection_rate


    def optimize_threshold(self):
        print(f"\n{'='*60}")
        print("Optimizing Confidence Threshold")
        print(f"{'='*60}")
        
        thresholds = np.arange(0.3, 0.95, 0.05)
        accuracies = []
        rejection_rates = []
        
        for threshold in thresholds:
            y_proba = self.svm_model.predict_proba(self.X_val_scaled)
            max_proba = np.max(y_proba, axis=1)
            
            y_pred = []
            for i, proba in enumerate(max_proba):
                if proba < threshold:
                    y_pred.append(6)
                else:
                    y_pred.append(self.svm_model.predict(self.X_val_scaled[i:i+1])[0])
            
            y_pred = np.array(y_pred)
            non_rejected_mask = y_pred != 6
            
            if np.sum(non_rejected_mask) > 0:
                acc = accuracy_score(self.y_val[non_rejected_mask], y_pred[non_rejected_mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
            
            rejection_rates.append(np.sum(y_pred == 6) / len(y_pred))
        
        # Find best threshold (accuracy >= 0.85 with minimum rejection)
        valid_thresholds = [(t, a, r) for t, a, r in zip(thresholds, accuracies, rejection_rates) if a >= 0.85]
        if valid_thresholds:
            best_threshold = min(valid_thresholds, key=lambda x: x[2])
            print(f"\nRecommended threshold: {best_threshold[0]:.2f}")
            print(f"Accuracy: {best_threshold[1]:.4%}")
            print(f"Rejection rate: {best_threshold[2]:.2%}")
            return best_threshold[0]
        else:
            print("\nNo threshold achieves 0.85 accuracy")
            best_idx = np.argmax(accuracies)
            print(f"Best available threshold: {thresholds[best_idx]:.2f}")
            print(f"Accuracy: {accuracies[best_idx]:.4%}")
            print(f"Rejection rate: {rejection_rates[best_idx]:.2%}")
            return thresholds[best_idx]


    def save_model(self, filename=None):
        if filename is None:
            filename = f'svm_model_{self.feature_type}.pkl'
        
        model_data = {
            'svm_model': self.svm_model,
            'scaler': self.scaler,
            'feature_type': self.feature_type,
            'confidence_threshold': self.confidence_threshold,
            'class_names': self.class_names
        }
        
        joblib.dump(model_data, filename)
        print(f"\nModel saved as '{filename}'")

    
def main():
        print("="*60)
        print("SVM CLASSIFIER TRAINING PIPELINE")
        print("="*60)
        
        feature_types = ['color', 'hog', 'both']
        results = {}
        
        for feat_type in feature_types:
            print(f"\n\n{'#'*60}")
            print(f"# TRAINING WITH {feat_type.upper()} FEATURES")
            print(f"{'#'*60}")
            
            svm_clf = SVMClassifier(feature_type=feat_type)
            
            svm_clf.load_data()
            svm_clf.prepare_data(test_size=0.2)

            #svm_clf.train_simple(kernel='rbf', C=20, gamma=0.01) // color
            #svm_clf.train_simple(kernel='rbf', C=20, gamma=0.01) // both
            #svm_clf.train_simple(kernel='rbf', C=20, gamma='scale') // both 
            #svm_clf.train_simple(kernel='rbf', C=15, gamma='scale') // both
            #svm_clf.train_simple(kernel='rbf', C=15, gamma=0.005) // both
            svm_clf.train_simple(kernel='rbf', C=10, gamma='scale')
            
            # Evaluate
            accuracy = svm_clf.evaluate()
            
            # Optimize threshold and evaluate with rejection
            best_threshold = svm_clf.optimize_threshold()
            post_acc, rejection_rate = svm_clf.evaluate_with_rejection(confidence_threshold=best_threshold)
            
            # Save model
            svm_clf.save_model()
            
            results[feat_type] = {
                'accuracy': accuracy,
                'post_accuracy': post_acc,
                'best_threshold': best_threshold,
                'rejection_rate': rejection_rate,
                'classifier': svm_clf
            }
        
        # Final comparison
        print(f"\n\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'='*60}")
        for feat_type, result in results.items():
            print(f"\n{feat_type.upper()} features:")
            print(f"  Raw Accuracy: {result['accuracy']:.4%}")
            print(f"  Post-threshold Accuracy: {result['post_accuracy']:.4%}")
            print(f"  Best Threshold: {result['best_threshold']:.2f}")
            print(f"  Rejection Rate: {result['rejection_rate']:.2%}")
        
        # Determine best model
        best_feat_type = max(results.keys(), key=lambda k: results[k]['post_accuracy'])
        print(f"\nBest performing features: {best_feat_type.upper()}")
        print(f"  Post-threshold Accuracy: {results[best_feat_type]['post_accuracy']:.4%}")


if __name__ == "__main__":
    main()
