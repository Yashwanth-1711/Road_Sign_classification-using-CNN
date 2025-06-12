import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import random

# --- Paths ---
MODEL_PATH = r"C:\My Files\Deep Learning project\road_sign_model_tuned.h5"
CLASS_NAMES_PATH = r"C:\My Files\Deep Learning project\class_names.pkl"
TEST_DATA_PATH = r"C:\My Files\Deep Learning project\Road Signs classification dataset\TEST"
IMG_SIZE = 160

# --- Load class names ---
def load_class_names(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# --- Load and preprocess test images ---
def load_test_images(folder_path, img_size, class_names):
    X, y = [], []
    for idx, class_name in enumerate(class_names):
        label_path = os.path.join(folder_path, str(idx))
        if not os.path.isdir(label_path):
            print(f"Warning: Directory '{label_path}' does not exist.")
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(idx)
    return np.array(X), np.array(y)

# --- Load model and class names ---
print("Loading model and class names...")
model = load_model(MODEL_PATH)
class_names = load_class_names(CLASS_NAMES_PATH)

# --- Load test data ---
print("Loading test images...")
X_test, y_test = load_test_images(TEST_DATA_PATH, IMG_SIZE, class_names)
X_test = preprocess_input(X_test.astype('float32'))

# --- Make predictions ---
print("Making predictions...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- Evaluation ---
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

# --- Per-Class Accuracy ---
print("\nPer-Class Accuracy:")
cm = confusion_matrix(y_test, y_pred)
cm_sum = np.sum(cm, axis=1)
cm_diag = np.diag(cm)
for idx, class_name in enumerate(class_names):
    acc = (cm_diag[idx] / cm_sum[idx]) * 100 if cm_sum[idx] > 0 else 0
    print(f"{class_name:<30}: {acc:.2f}%")

# --- Confusion Matrix ---
print("Plotting Confusion Matrix...")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(10, 8))
disp.plot(cmap='Oranges', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()

# --- Save Report and Confusion Matrix ---
plt.savefig("confusion_matrix.png")
with open("classification_report.txt", "w") as f:
    f.write(report)
print("Saved 'classification_report.txt' and 'confusion_matrix.png'")

plt.show()

# --- Display sample predictions ---
print("\nDisplaying sample predictions...")
sample_indices = random.sample(range(len(X_test)), 6)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(sample_indices):
    # Original image before preprocessing
    orig_img = X_test[idx]  # this is preprocessed, we'll revert it
    img = orig_img.copy()
    img = img.astype(np.uint8)  # reverse preprocess only for display

    true_label = class_names[y_test[idx]]
    pred_label = class_names[y_pred[idx]]

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()
