import cv2
import numpy as np

def trim_video(input_file, output_file, start_time, end_time):
    """Trims a video from start_time to end_time seconds.

    Args:
        input_file (str): Path to the input video file.
        output_file (str): Path to the output video file.
        start_time (float): Start time of the trimmed video in seconds.
        end_time (float): End time of the trimmed video in seconds.
    """

    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Replace 'mp4v' with your desired codec
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= start_frame and frame_count <= end_frame:
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()

    cv2.destroyAllWindows()

# Example usage:
input_video = '1_12947945544.mp4'
output_video = '1_12_trimmed_video.mp4'
start_time = 56  # seconds
end_time = 66  # seconds

trim_video(input_video, output_video, start_time, end_time)