import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import json

# Step 1: Data Loading and Preprocessing

# Define the base path
base_path = r'path to the post processed dataset'

# Initialize lists to store sequences and labels
sequences = []
labels = []

# Define the expected feature columns
feature_columns = [
    'left_stride_length',
    'right_stride_length',
    'gait_symmetry',
    'left_stride_variability',
    'right_stride_variability'
]

# **Initialize counters for each label**
label_counts = {'drunk': 0, 'normal': 0}
max_files_per_label = 200  # Limit to 200 files per label

# Recursively search for CSV files and extract sequences and labels
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file == 'stride_lengths_with_symmetry_and_variability.csv':
            csv_path = os.path.join(root, file)

            # Extract label from the folder name (assuming 'drunk' or 'normal')
            label_folder = os.path.basename(os.path.dirname(os.path.dirname(csv_path))).lower()
            if label_folder == 'drunk':
                label = 1
            elif label_folder == 'normal':
                label = 0
            else:
                print(f"Unknown label folder '{label_folder}' in path: {csv_path}")
                continue  # Skip if the label is not 'drunk' or 'normal'

                # **Check if we've already processed 200 files for this label**
            if label_counts[label_folder] >= max_files_per_label:
                continue  # Skip this file

            # Read the CSV file
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace

                # Print columns for debugging
                # print(f"Columns in {csv_path}: {df.columns.tolist()}")

                # Check if required columns are present
                if all(col in df.columns for col in feature_columns):
                    # Handle missing values
                    df = df.fillna(0)

                    # Extract features as a sequence (time_steps, features)
                    sequence = df[feature_columns].values

                    # Append to lists
                    sequences.append(sequence)
                    labels.append(label)

                    # **Increment the counter for this label**
                    label_counts[label_folder] += 1

                    # **Check if we've reached the limit for both labels**
                    if label_counts['drunk'] >= max_files_per_label and label_counts['normal'] >= max_files_per_label:
                        break  # Exit the inner loop

                else:
                    print(f"Missing required columns in {csv_path}")
                    print(f"Available columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
    else:
        # Continue if the inner loop wasn't broken
        continue
    # Break the outer loop if we've reached the limit for both labels
    break

# Convert lists to numpy arrays
labels_array = np.array(labels)

# Check if data is loaded
if len(sequences) == 0:
    print("No data loaded. Please check the file paths and column names.")
    exit()

print("Data loaded successfully.")
print(f"Number of sequences: {len(sequences)}")
print(f"Label counts: {label_counts}")
print(f"Labels shape: {labels_array.shape}")

# Step 2: Normalize the Sequences

# Flatten all sequences to fit the scaler and then reshape back
all_data = np.concatenate(sequences, axis=0)
scaler = MinMaxScaler()
scaler.fit(all_data)

# Normalize each sequence
normalized_sequences = [scaler.transform(seq) for seq in sequences]

# **Save the scaler here**
scaler_filename = 'scaler.save'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")

# Step 3: Pad the Sequences and Create a Mask

# Determine the maximum sequence length
max_sequence_length = max(seq.shape[0] for seq in normalized_sequences)
print(f"Maximum sequence length: {max_sequence_length}")

# Pad sequences with a mask value (e.g., -1.0)
mask_value = -1.0
padded_sequences = pad_sequences(
    normalized_sequences,
    maxlen=max_sequence_length,
    dtype='float32',
    padding='post',
    value=mask_value
)

print("Padded sequences shape:", padded_sequences.shape)

# **Save preprocessing parameters here**
preprocessing_params = {
    'max_sequence_length': max_sequence_length,
    'mask_value': mask_value,
    'feature_columns': feature_columns
}

with open('preprocessing_params.json', 'w') as f:
    json.dump(preprocessing_params, f)
print("Preprocessing parameters saved to preprocessing_params.json")

# Step 4: Split the Data into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels_array, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Step 5: Define the LSTM Model with Masking

from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout

num_features = len(feature_columns)

# Build the model
model = Sequential()
model.add(Masking(mask_value=mask_value, input_shape=(max_sequence_length, num_features)))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Step 6: Train the Model

# Convert labels to float32 for compatibility
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16
)

# Step 7: Evaluate the Model

# Define the path where you want to save the model
model_save_path = 'trained_lstm_model.keras'

# Save the model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Optional: Plot training history (Requires matplotlib)
import matplotlib.pyplot as plt

# Plot accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype('int32').flatten()

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Optionally, print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Drunk']))

# Plot the confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Drunk'], yticklabels=['Normal', 'Drunk'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
