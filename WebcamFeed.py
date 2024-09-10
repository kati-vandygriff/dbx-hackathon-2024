import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:/Users/ss969/Downloads/helmet_model_reassembled_v1.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    label = 'Helmet' if prediction[0][0] > 0.5 else 'No Helmet'

    # Display the result
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Helmet Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
