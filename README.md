🚦 CNN-Based Road Signs Classification | Deep Learning Project
This project focuses on building a Convolutional Neural Network (CNN) to accurately classify traffic and road signs into one of 30 predefined categories (e.g., speed limit, stop sign, yield). It's an end-to-end deep learning solution involving data preprocessing, model building, evaluation, optimization, and deployment with an interactive UI.

📁 Dataset Overview
Images: Real-world road sign images.

Classes: 30 different road sign categories.

Labels: Provided in an Excel file.

Split: Training, validation, and test sets for robust model performance.

⚙️ Project Workflow
🔹 Step 1: Data Collection & Preprocessing

Downloaded and extracted images.

Resized to a consistent shape (e.g., 64x64).

Normalized pixel values (0–1 scale).

Encoded labels using one-hot encoding.

Applied data augmentation (rotation, flip, zoom) for improved generalization.

🔹 Step 2: CNN Architecture Design

Input Layer: Accepts 64x64 RGB images.

Conv Layers: Extract spatial features.

Pooling Layers: Reduce dimensions via max pooling.

Fully Connected Layers: Interpret features and classify.

Activation Functions: ReLU (hidden layers), Softmax (output layer).

🔹 Step 3: Model Compilation & Training

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Data split: 70% Training, 15% Validation, 15% Testing.

Used Early Stopping & Model Checkpoints to prevent overfitting.

🔹 Step 4: Model Evaluation

Evaluated on the test dataset.

Metrics: Accuracy, Precision, Recall, F1-Score.

Visualized results with a Confusion Matrix and class-wise performance charts.

Tested model on new, unseen images to validate real-world effectiveness.

🔹 Step 5: Hyperparameter Tuning & Optimization

Performed Grid Search and Random Search for optimal parameters (learning rate, filter size, etc.).

Regularized using Dropout Layers.

Experimented with deeper architectures to boost performance.

🔹 Step 6: Streamlit-Based User Interface

Developed a clean and interactive Streamlit app.

Users can upload images and get real-time predictions with confidence scores and class labels.

Dashboard includes model performance metrics and visualization plots.

✅ Final Deliverables
🎯 Trained CNN Model (ready for deployment)

📊 Performance Reports: Accuracy, Precision, Recall, F1-score

📈 Visualizations: Training/validation curves, confusion matrix

🖼️ Streamlit UI: Predicts road sign class from image uploads

