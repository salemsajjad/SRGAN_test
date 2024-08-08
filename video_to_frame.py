import cv2
import os

def save_frame(frame, directory, filename):
    """Saves a frame as a BMP image in the specified directory."""
    # resized_frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_CUBIC)
    path = os.path.join(directory, filename)
    cv2.imwrite(path, frame)
    print(f"Frame saved as {path}")

# Replace 'your_video_path' with the actual path to your video file
video_path = '1_12_trimmed_video.mp4'
save_directory = '1_12_saved_frames'  # Replace with your desired directory

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Video', frame)

    key = cv2.waitKey(10) & 0xFF
    # if key == ord('s'):
    filename = f"frame_{frame_count}.bmp"
    save_frame(frame, save_directory, filename)
    frame_count += 1
    if key == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()