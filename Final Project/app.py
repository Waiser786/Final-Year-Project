from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import joblib
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask
from flask_cors import CORS
from flask import send_from_directory, render_template, request, jsonify
from werkzeug.utils import secure_filename



app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to serve the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Configuration Parameters
BASE_CONFIDENCE_THRESHOLD = 0.5
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Load the trained LSTM model
model = load_model('trained_lstm_model.keras')
print("LSTM Model loaded successfully.")

# Load the scaler
scaler = joblib.load('scaler.save')
print("Scaler loaded successfully.")

# Load preprocessing parameters
with open('preprocessing_params.json', 'r') as f:
    preprocessing_params = json.load(f)
max_sequence_length = preprocessing_params['max_sequence_length']
mask_value = preprocessing_params['mask_value']
feature_columns = preprocessing_params['feature_columns']
print("Preprocessing parameters loaded successfully.")

# Helper functions for calculating stride length, gait symmetry, and variability
def calculate_stride_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_gait_symmetry(left_stride, right_stride):
    if left_stride + right_stride == 0 or np.isnan(left_stride) or np.isnan(right_stride):
        return np.nan
    return (2 * (left_stride - right_stride) / (left_stride + right_stride)) * 100

def calculate_stride_variability(stride_lengths):
    stride_lengths = stride_lengths.dropna()
    mean_stride = stride_lengths.mean()
    std_stride = stride_lengths.std()
    if mean_stride != 0:
        return (std_stride / mean_stride) * 100
    return np.nan


def display_keypoints(image, detection_bbox, keypoints, frame_idx):
    ankle_coords = {'left_ankle': None, 'right_ankle': None}

    # Define pairs of keypoints to connect with lines (skeleton structure)
    skeleton_connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle")
    ]

    # Dictionary to store the coordinates of detected keypoints
    keypoint_coords = {}

    # Draw keypoints
    for kp_idx, kp in enumerate(keypoints):
        x, y, conf = kp
        kp_name = KEYPOINT_NAMES[kp_idx]
        print(f"Frame {frame_idx}, Keypoint {kp_name}: x={x}, y={y}, conf={conf}")

        if conf < BASE_CONFIDENCE_THRESHOLD:
            continue

        # Draw the keypoint on the image
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green color for keypoints
        keypoint_coords[kp_name] = (int(x), int(y))

        # Store ankle coordinates for later processing
        if kp_name == "left_ankle":
            ankle_coords['left_ankle'] = (x, y)
        elif kp_name == "right_ankle":
            ankle_coords['right_ankle'] = (x, y)

    # Draw skeleton connections
    for joint_pair in skeleton_connections:
        kp1_name, kp2_name = joint_pair
        if kp1_name in keypoint_coords and kp2_name in keypoint_coords:
            cv2.line(image, keypoint_coords[kp1_name], keypoint_coords[kp2_name], (255, 0, 0),
                     2)  # Blue color for skeleton lines

    return image, ankle_coords


def process_video(video_path, output_csv, output_video_path, model_path='yolov8s-pose.pt'):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices for evenly spaced frames
    selected_frames = np.linspace(0, total_frames - 1, 10, dtype=int)
    print(f"Selected frames: {selected_frames}")

    # Prepare video writer to save processed frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Use video FPS or default to 30

    # Since we have only 10 frames, set the output video FPS accordingly
    output_fps = 1  # Display each frame for 1 second (adjust as needed)

    print(f"Video Writer Initialized with frame size: {frame_width}x{frame_height}, FPS: {output_fps}")

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'H264'),
        output_fps,
        (frame_width, frame_height)
    )

    stride_data = []

    # Read and process only the selected frames
    for frame_number in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_number}.")
            continue

        try:
            results = model(frame, conf=BASE_CONFIDENCE_THRESHOLD)
            if len(results[0].boxes) == 0:
                print(f"No detections in frame {frame_number}.")
                out.write(frame)  # Save unmodified frame
                stride_data.append({
                    'frame': frame_number,
                    'left_ankle_x': np.nan,
                    'left_ankle_y': np.nan,
                    'right_ankle_x': np.nan,
                    'right_ankle_y': np.nan
                })
                continue

            print(f"Detections found in frame {frame_number}.")

            detection_bbox = results[0].boxes.xyxy[0].cpu().numpy()
            keypoints_data = results[0].keypoints.xy[0].cpu().numpy()
            keypoints_conf = results[0].keypoints.conf[0].cpu().numpy().reshape(-1, 1)
            keypoints = np.hstack((keypoints_data, keypoints_conf))
            frame, ankle_coords = display_keypoints(frame, detection_bbox, keypoints, frame_number)

            # Save the frame with keypoints drawn
            out.write(frame)
            print(f"Frame {frame_number} processed and written.")

            # Collect stride data
            stride_data.append({
                'frame': frame_number,
                'left_ankle_x': ankle_coords['left_ankle'][0] if ankle_coords['left_ankle'] else np.nan,
                'left_ankle_y': ankle_coords['left_ankle'][1] if ankle_coords['left_ankle'] else np.nan,
                'right_ankle_x': ankle_coords['right_ankle'][0] if ankle_coords['right_ankle'] else np.nan,
                'right_ankle_y': ankle_coords['right_ankle'][1] if ankle_coords['right_ankle'] else np.nan
            })
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")

    cap.release()
    out.release()
    print("Video processing completed. Video writer released.")

    # Save stride data to CSV
    df_stride = pd.DataFrame(stride_data)
    df_stride.to_csv(output_csv, index=False)
    return output_csv



def process_single_stride_file(input_csv, output_csv):
    # Load the CSV file
    if not os.path.isfile(input_csv):
        print(f"{input_csv} not found. Skipping.")
        return

    df = pd.read_csv(input_csv)

    # Initialize stride lengths with NaN
    df['left_stride_length'] = np.nan
    df['right_stride_length'] = np.nan

    # Iterate through the DataFrame to compute stride lengths for left and right ankles
    for i in range(len(df) - 1):
        # Calculate left stride length between current frame and the next one
        if not (np.isnan(df.loc[i, 'left_ankle_x']) or np.isnan(df.loc[i + 1, 'left_ankle_x']) or
                np.isnan(df.loc[i, 'left_ankle_y']) or np.isnan(df.loc[i + 1, 'left_ankle_y'])):
            df.loc[i, 'left_stride_length'] = calculate_stride_length(
                df.loc[i, 'left_ankle_x'], df.loc[i, 'left_ankle_y'],
                df.loc[i + 1, 'left_ankle_x'], df.loc[i + 1, 'left_ankle_y']
            )

        # Calculate right stride length between current frame and the next one
        if not (np.isnan(df.loc[i, 'right_ankle_x']) or np.isnan(df.loc[i + 1, 'right_ankle_x']) or
                np.isnan(df.loc[i, 'right_ankle_y']) or np.isnan(df.loc[i + 1, 'right_ankle_y'])):
            df.loc[i, 'right_stride_length'] = calculate_stride_length(
                df.loc[i, 'right_ankle_x'], df.loc[i, 'right_ankle_y'],
                df.loc[i + 1, 'right_ankle_x'], df.loc[i + 1, 'right_ankle_y']
            )

    # Calculate Gait Symmetry for each row
    df['gait_symmetry'] = df.apply(
        lambda row: calculate_gait_symmetry(row['left_stride_length'], row['right_stride_length']), axis=1
    )

    # Calculate Stride Variability for left and right strides
    left_stride_variability = calculate_stride_variability(df['left_stride_length'])
    right_stride_variability = calculate_stride_variability(df['right_stride_length'])

    # Add stride variability to the DataFrame
    df['left_stride_variability'] = left_stride_variability
    df['right_stride_variability'] = right_stride_variability

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Stride lengths, gait symmetry, and stride variability calculated and saved to {output_csv}")
    return output_csv

def predict_gait_label(csv_path):
    # Step 2: Load and preprocess the new CSV file
    try:
        df_new = pd.read_csv(csv_path)
        df_new.columns = df_new.columns.str.strip()  # Remove any leading/trailing whitespace
        print(f"Columns in the new CSV file: {df_new.columns.tolist()}")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None, None

    # Check if required columns are present
    if not all(col in df_new.columns for col in feature_columns):
        missing_cols = [col for col in feature_columns if col not in df_new.columns]
        print(f"Missing required columns: {missing_cols}")
        return None, None

    # Handle missing values (if any)
    df_new = df_new.fillna(0)

    # Extract features as a sequence (time_steps, features)
    sequence_new = df_new[feature_columns].values
    print(f"Original sequence shape: {sequence_new.shape}")

    # Normalize the sequence
    sequence_new_normalized = scaler.transform(sequence_new)

    # Pad the sequence
    sequence_new_padded = pad_sequences(
        [sequence_new_normalized],  # Input should be a list of sequences
        maxlen=max_sequence_length,
        dtype='float32',
        padding='post',
        value=mask_value
    )

    print(f"Padded sequence shape: {sequence_new_padded.shape}")

    # Step 3: Make the prediction
    prediction_probs = model.predict(sequence_new_padded)
    drunk_probability = float(prediction_probs[0][0])
    prediction = (prediction_probs > 0.5).astype('int32').flatten()[0]

    # Interpret the prediction
    label_map = {0: 'Normal', 1: 'Drunk'}
    predicted_label = label_map[prediction]

    # Calculate final probability and convert to percentage format
    if predicted_label == 'Drunk':
        final_probability = drunk_probability * 100  # Convert to percentage
    else:
        final_probability = (1 - drunk_probability) * 100  # Convert to percentage

    # Format the final probability to one decimal place
    final_probability = round(final_probability, 1)

    print(f"Predicted label: {predicted_label}")
    print(f"Prediction probability: {final_probability:.1f}%")

    return predicted_label, final_probability


# Route to handle the video processing
@app.route('/process', methods=['POST'])
def process():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video = request.files['video']
    video_filename = secure_filename(video.filename)
    video_path = os.path.join('uploads', video_filename)
    video.save(video_path)

    output_csv = 'ankle_coordinates.csv'
    stride_output_csv = 'stride_lengths_with_symmetry_and_variability.csv'
    processed_video_filename = 'processed_' + video_filename
    processed_video_path = os.path.join('uploads', processed_video_filename)

    try:
        # Step 1: Process video to extract ankle coordinates and save processed video
        process_video(video_path, output_csv, processed_video_path)

        # Step 2: Calculate stride lengths, gait symmetry, and variability
        process_single_stride_file(output_csv, stride_output_csv)

        # Step 3: Make a prediction
        predicted_label, prediction_probability = predict_gait_label(stride_output_csv)
        if predicted_label is None:
            return jsonify({'error': 'Error during prediction'}), 500
    except Exception as e:
        app.logger.error(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'message': 'Video processed successfully',
        'csv': stride_output_csv,
        'predicted_label': predicted_label,
        'prediction_probability': prediction_probability,
        'processed_video': processed_video_filename
    }), 200

# Route to serve files from the 'uploads' directory
@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        app.logger.debug(f"File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(UPLOAD_FOLDER, filename, mimetype='video/mp4')


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
