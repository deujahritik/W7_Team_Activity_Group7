import cv2

# Load the face detection model
face_cascade = cv2.CascadeClassifier('path/to/haar/cascade/xml/file')

# Initialize the video stream from the Raspberry Pi camera module
cap = cv2.VideoCapture(0)

while True:
    # Read the video frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the video frame
    cv2.imshow('Face Detection', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
