import os
import zipfile
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# === Step 1: Unzip the files ===
def unzip_file(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

unzip_file("C:/python/prodigy_internship/project2/train.zip", "data/train")
unzip_file("C:/python/prodigy_internship/project2/test1.zip", "data/test")

# === Step 2: Image Preprocessing ===
def preprocess_image(image_path, size=(64, 64)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray.flatten()

# === Step 3: Load Training Data ===
def load_training_data(folder):
    features = []
    labels = []

    for filename in tqdm(os.listdir(folder)):
        if filename.endswith('.jpg'):
            label = 0 if 'cat' in filename.lower() else 1  # 0: cat, 1: dog
            img_path = os.path.join(folder, filename)
            try:
                features.append(preprocess_image(img_path))
                labels.append(label)
            except:
                continue  # skip unreadable images

    return np.array(features), np.array(labels)

X, y = load_training_data("data/train/train")

# Limit to 2500 cats and 2500 dogs
cat_indices = np.where(y == 0)[0][:2500]
dog_indices = np.where(y == 1)[0][:2500]

selected_indices = np.concatenate((cat_indices, dog_indices))
X = X[selected_indices]
y = y[selected_indices]


# === Step 4: Split Data and Train SVM ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training SVM model...")
model = SVC(kernel='linear', verbose=True)
model.fit(X_train, y_train)

# === Step 5: Validation Accuracy ===
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# === Step 6: Load Test Data ===
def load_test_data(folder):
    test_features = []
    test_ids = []

    for filename in tqdm(os.listdir(folder)):
        if filename.endswith('.jpg'):
            img_id = int(filename.split('.')[0])  # get number from "123.jpg"
            img_path = os.path.join(folder, filename)
            try:
                test_features.append(preprocess_image(img_path))
                test_ids.append(img_id)
            except:
                continue

    return np.array(test_features), test_ids

X_test, test_ids = load_test_data("data/test/test1")

# === Step 7: Predict on Test Data ===
print("Predicting on test set...")
y_test_pred = model.predict(X_test)

# === Step 8: Save Submission File ===
submission = pd.DataFrame({
    'id': test_ids,
    'label': y_test_pred
})

submission = submission.sort_values(by="id")  # sort by id for submission consistency
submission.to_csv("svm_submission.csv", index=False)

print("Submission file saved as svm_submission.csv")
