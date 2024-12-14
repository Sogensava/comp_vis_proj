import cv2
import os

name_list = ['ant','duck','misato','referee']

for name in name_list:
    # Directory containing the BMP images
    image_folder = fr"C:\Users\Sogensawa\Desktop\AHAHABRA\Basladik_Heralde\python_ws\comp_vis_project_dataset\{name}" # Replace with your folder path
    output_video = f"{name}.mp4"  # Name of the output video file

    # Parameters for the video
    frame_rate = 30  # Frames per second

    # Get list of images in the directory
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".bmp")]
    if not images:
        print("No BMP files found in the specified directory.")
        exit()

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec if needed
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Write images to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer object
    video.release()
    print(f"Video saved as {output_video}")