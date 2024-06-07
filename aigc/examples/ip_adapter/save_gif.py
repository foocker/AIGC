import cv2
import os

def save_images_as_mp4(image_folder, output_path, fps=2):
    """
    Save a sequence of images as an MP4 video using OpenCV.

    Parameters:
    - image_folder: Folder containing the images.
    - output_path: Path to save the MP4 video.
    - fps: Frames per second.
    """
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_folder, file_name)
            img = cv2.imread(file_path)
            img = cv2.resize()
            if img is not None:
                images.append(img)
    
    if not images:
        print("No images found in the specified folder.")
        return

    # 获取图像的高度和宽度
    height, width, layers = images[0].shape
    size = (width, height)

    # 创建一个视频写入器对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for image in images:
        out.write(image)

    out.release()
    print(f"Video saved to {output_path}")

# 示例使用
image_folder = 'results'
output_path = 'output.mp4'
save_images_as_mp4(image_folder, output_path)

