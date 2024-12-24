import os
import random
import shutil

dataset_dir = r"D:\MyWork\ComputerVisionMaskDetector\MaskDetectorData\YeniBuldumLabelli\together"  
train_dir = r"D:\MyWork\ComputerVisionMaskDetector\MaskDetectorData\YeniBuldumLabelli\Train"   
val_dir = r"D:\MyWork\ComputerVisionMaskDetector\MaskDetectorData\YeniBuldumLabelli\Val"        

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

xml_files = [f for f in os.listdir(dataset_dir) if f.endswith('.xml')]

random.seed(42)  
random.shuffle(xml_files)

total = len(xml_files)
val_size = int(total * 0.1)  
train_size = total - val_size

val_files = xml_files[:val_size]
train_files = xml_files[val_size:]

for xml_file in train_files:
    shutil.copy(os.path.join(dataset_dir, xml_file), os.path.join(train_dir, xml_file))
    image_file = xml_file.replace('.xml', '.jpg')  
    shutil.copy(os.path.join(dataset_dir, image_file), os.path.join(train_dir, image_file))

for xml_file in val_files:
    shutil.copy(os.path.join(dataset_dir, xml_file), os.path.join(val_dir, xml_file))
    image_file = xml_file.replace('.xml', '.jpg') 
    shutil.copy(os.path.join(dataset_dir, image_file), os.path.join(val_dir, image_file))

print(f"Dataset split completed: {train_size} training and {val_size} validation files.")
