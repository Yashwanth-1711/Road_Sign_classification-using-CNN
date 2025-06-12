import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras_tuner import RandomSearch

# --- Load labels ---
label_csv_path = r"C:\\My Files\\Deep Learning project\\Road Signs classification dataset\\labels.csv"
label_df = pd.read_csv(label_csv_path)
label_df = label_df.sort_values(by='ClassId').reset_index(drop=True)
class_ids = label_df['ClassId'].astype(str).tolist()
class_names = label_df['Name'].tolist()

# --- Load images function ---
def load_images(folder_path, img_size):
    X, y = [], []
    for label_str in class_ids:
        label_path = os.path.join(folder_path, label_str)
        if not os.path.isdir(label_path):
            print(f"Warning: Folder {label_path} not found!")
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(class_ids.index(label_str))
    return np.array(X), np.array(y)

# --- Load dataset ---
data_path = r"C:\\My Files\\Deep Learning project\\Road Signs classification dataset\\DATA"
img_size = 160
X, y = load_images(data_path, img_size)
num_classes = len(class_names)

X = preprocess_input(X.astype('float32'))

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
for train_idx, val_idx in sss.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

# --- Train and Evaluate Model BEFORE Tuning ---
print("\\nTraining model BEFORE tuning for comparison...")

base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False

inputs = Input(shape=(img_size, img_size, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-3))(x)

model_before_tuning = Model(inputs, outputs)
model_before_tuning.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_before = model_before_tuning.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=10,
    callbacks=[
        EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ],
    verbose=1
)

# --- Evaluation Before Tuning ---
y_val_preds_before = model_before_tuning.predict(X_val)
y_val_pred_classes_before = np.argmax(y_val_preds_before, axis=1)

print("\\nClassification Report (Before Tuning):")
print(classification_report(y_val, y_val_pred_classes_before, target_names=class_names))

cm_before = confusion_matrix(y_val, y_val_pred_classes_before)
disp_before = ConfusionMatrixDisplay(confusion_matrix=cm_before, display_labels=class_names)
plt.figure(figsize=(10, 8))
disp_before.plot(cmap='Oranges', xticks_rotation=45)
plt.title("Confusion Matrix (Before Tuning)")
plt.show()

# --- Build model function for tuner ---
def build_model(hp):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    
    unfreeze_layers = hp.Int('unfreeze_layers', min_value=10, max_value=20, step=5)
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    inputs = Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    dropout_rate = hp.Float('dropout', 0.5, 0.7, step=0.1)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-3))(x)

    model = Model(inputs, outputs)
    lr_choice = hp.Choice('lr', [1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=lr_choice),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Create tuner ---
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='tuner_logs',
    project_name='road_signs_tune'
)

tuner.search_space_summary()

# --- Run hyperparameter search ---
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=10,
    callbacks=[
        EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ]
)

# --- Get best hyperparameters and build best model ---
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\\nBest Hyperparameters:")
print(best_hp.values)

model = tuner.hypermodel.build(best_hp)

# --- Train best model ---
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=10,
    callbacks=callbacks
)

# --- Evaluate model after tuning ---
y_val_preds = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_preds, axis=1)

print("\\n🔍 Classification Report (After Tuning):")
print(classification_report(y_val, y_val_pred_classes, target_names=class_names))

cm = confusion_matrix(y_val, y_val_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (After Tuning)")
plt.show()

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\\nFinal Validation Accuracy: {val_acc * 100:.2f}%")

# --- Save model and class names ---
model.save(r"C:\My Files\Deep Learning project\road_sign_model_tuned.h5")
with open("class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

# --- Plot training history ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss (After Tuning)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy (After Tuning)')
plt.legend()

plt.tight_layout()
plt.show()