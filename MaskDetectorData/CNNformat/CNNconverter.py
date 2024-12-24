import os
from PIL import Image

img_dir = r"D:\MyWork\ComputerVisionMaskDetector\MaskDetectorData\CNNformat\Train\images"
yolo_label_dir = r"D:\MyWork\ComputerVisionMaskDetector\MaskDetectorData\CNNformat\Train\labels"
output_dir = r"D:\MyWork\ComputerVisionMaskDetector\MaskDetectorData\CNNformat\degismis\Labels"

class_names = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_weared_incorrect"
}

def convert_yolo_to_cnn(img_dir, yolo_label_dir, output_dir):
    for label_file in os.listdir(yolo_label_dir):
        if label_file.endswith(".txt"):
            img_name = os.path.splitext(label_file)[0]
            img_path_jpg = os.path.join(img_dir, f"{img_name}.jpg")
            img_path_png = os.path.join(img_dir, f"{img_name}.png")

            if os.path.exists(img_path_jpg):
                img_path = img_path_jpg
            elif os.path.exists(img_path_png):
                img_path = img_path_png
            else:
                print(f"Image not found for {label_file}")
                continue

            label_path = os.path.join(yolo_label_dir, label_file)
            with open(label_path, 'r') as f:
                labels = f.readlines()

            img = Image.open(img_path)
            width, height = img.size

            for idx, label in enumerate(labels):
                data = label.strip().split()
                class_id, x_center, y_center, w, h = map(float, data)

                x_min = int((x_center - w / 2) * width)
                y_min = int((y_center - h / 2) * height)
                x_max = int((x_center + w / 2) * width)
                y_max = int((y_center + h / 2) * height)

                roi = img.crop((x_min, y_min, x_max, y_max))
                
                if roi.mode == "RGBA":
                    roi = roi.convert("RGB")

                class_name = class_names.get(int(class_id), "unknown")
                class_folder = os.path.join(output_dir, class_name)
                os.makedirs(class_folder, exist_ok=True)
                roi.save(os.path.join(class_folder, f'{img_name}_{idx}.jpg'))


convert_yolo_to_cnn(img_dir, yolo_label_dir, output_dir)
