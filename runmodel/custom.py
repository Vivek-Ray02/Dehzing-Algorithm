import cv2
from image_dehazer import image_dehazer

# Initialize the dehazer
dehazer = image_dehazer(airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                        regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=False)

file_path = 'videosample.mp4'  # For a video

# Check if the input is an image or a video
is_video = False
if file_path.endswith('.mp4') or file_path.endswith('.avi') or file_path.endswith('.mov'):
    is_video = True
    cap = cv2.VideoCapture(file_path)
else:
    # Read the image
    img = cv2.imread(file_path)

if is_video:
    # Process the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the dehazing model to the frame
        dehazed_frame, _ = dehazer.remove_haze(frame)

        # Display the dehazed frame
        cv2.imshow('Dehazed Video', dehazed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
else:
    # Apply the dehazing model to the image
    dehazed_img, _ = dehazer.remove_haze(img)

    # Display the dehazed image
    cv2.imshow('Dehazed Image', dehazed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()