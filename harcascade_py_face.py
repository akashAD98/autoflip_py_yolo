import cv2

# Load the video file
video = cv2.VideoCapture("icegov.mp4")

# Get the video's dimensions
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

size = (width, height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('icegovop.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

# Initialize the face detector
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Loop through the frames of the video
while True:
    # Read a frame from the video
    success, frame = video.read()
    result.write(frame)

    # Break out of the loop if we've reached the end of the video
    if not success:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray)

    # If there are faces, adjust the size of the frame
    if len(faces) > 0:
        # Get the largest face
        face = max(faces, key=lambda x: x[2] * x[3])

        # Get the face's bounding box
        x, y, w, h = face

        # Determine the desired size for the face
        face_size = 200

        # Calculate the aspect ratio of the face
        aspect_ratio = w / h

        # Calculate the width and height of the resized frame
        new_width = int(face_size * aspect_ratio)
        new_height = int(face_size / aspect_ratio)

        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height))
        #out.write(frame)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()
result.release()

# Close all open windows
cv2.destroyAllWindows()
