import cv2
from ICA import get_colorchannels
from sklearn.decomposition import FastICA

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera, change it to the desired video source if necessary

# Initialize FastICA with the number of components you want
n_components = 3  # You can change this number as needed
ica = FastICA(n_components=n_components, random_state=0)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    if not ret:
        break

    # Get the color channels of the frame
    color_channels = get_colorchannels(frame)

    # Calculate the ICA of each color channel and print to terminal
    red_channel_reshaped = color_channels[2].reshape(-1, 1)  # Reshape to a 1D array
    independent_components = ica.fit_transform(red_channel_reshaped)
    #print(independent_components)
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with face detections
    cv2.imshow('Face Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

