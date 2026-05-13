# real-time material classification using the best trained model
# run: python src/realtime_app.py

import os
import sys
import cv2
import numpy as np
import joblib

# add src/ to path so we can import feature_extraction directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_features

# -- configuration --
# paths are relative to this script so it works on any machine
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

# confidence threshold — predictions below this are classified as unknown
CONFIDENCE_THRESHOLD = 0.5

# class names including unknown
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash", "unknown"]

# display color per class in bgr format
CLASS_COLORS = {
    "glass"    : (255, 200, 0),
    "paper"    : (0, 255, 0),
    "cardboard": (0, 165, 255),
    "plastic"  : (255, 0, 255),
    "metal"    : (200, 200, 200),
    "trash"    : (0, 0, 255),
    "unknown"  : (128, 128, 128),
}


# -- load model and scaler --
def load_model():
    # load the fitted scaler saved by feature_extraction.py
    scaler_path = os.path.join(FEATURES_DIR, "scaler.pkl")
    model_path  = os.path.join(MODELS_DIR, "svm_model.pkl")

    # check both files exist before trying to load
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            f"scaler not found: {scaler_path}\n"
            "run feature_extraction.py first."
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"model not found: {model_path}\n"
            "download svm_model.pkl from google drive and place it in models/"
        )

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)

    print("Model and scaler loaded successfully")
    return scaler, model


# -- rejection mechanism --
def classify_with_rejection(feature_vector, scaler, model):
    # scale the feature vector using the same scaler used during training
    scaled_vector = scaler.transform([feature_vector])

    # get confidence scores for each of the 6 classes
    probabilities = model.predict_proba(scaled_vector)[0]

    # find the highest confidence score and which class it belongs to
    max_confidence = np.max(probabilities)
    predicted_class = np.argmax(probabilities)

    # if confidence is too low, reject as unknown
    if max_confidence < CONFIDENCE_THRESHOLD:
        return "unknown", max_confidence

    # otherwise return the predicted class name and its confidence
    return CLASS_NAMES[predicted_class], max_confidence


# -- display --
def draw_result(frame, label, confidence):
    # get the display color for this class
    color = CLASS_COLORS[label]

    # build the text to display — class name and confidence percentage
    text = f"{label.upper()}  {confidence*100:.1f}%"

    # draw a filled black rectangle at the top of the frame as text background
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)

    # draw the classification result on top of the rectangle
    cv2.putText(frame, text, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

    return frame


# -- main app --
def main():
    # load model and scaler once at startup
    scaler, model = load_model()

    # open the default camera (0 = built-in webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera")
        return

    print("Camera opened. Press Q to quit.")

    while True:
        # read one frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame")
            break

        # extract features from the current frame
        feature_vector = extract_features(frame)

        # classify with rejection mechanism
        label, confidence = classify_with_rejection(feature_vector, scaler, model)

        # draw the result on the frame
        frame = draw_result(frame, label, confidence)

        # show the frame in a window
        cv2.imshow("MSI - Material Stream Identification", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()