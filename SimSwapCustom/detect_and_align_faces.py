import os
import cv2
from tqdm import tqdm
from face_detect_crop_single import Face_detect_crop

input_data_path = '/content/lfw_funneled'
aligned_data_path = '/content/lfw_funneled_aligned'
dimension = 256

print("Load InsightFace antelopev2 ONNX models")
app = Face_detect_crop(name='antelopev2', 
  root='/content/SimSwap/insightface_func/models') 
app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))

total_processed = 0
faces_found = 0
no_face_found = []

all_image_paths = []

print("Find all the original face images")
for root, dirs, files in os.walk(input_data_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_image_paths.append(os.path.join(root, file))

print(f"{len(all_image_paths)} total face images.")

print("Cycle through and align all face images.")
print(f"Write out to new directory {aligned_data_path}")
for img_path in tqdm(all_image_paths):
    relative_path = os.path.relpath(img_path, input_data_path)
    
    out_path = os.path.join(aligned_data_path, relative_path)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    try:
        results = app.get(img, crop_size=dimension)
        
        if results is not None:
            aligned_images, _ = results

            cv2.imwrite(out_path, aligned_images[0])
            faces_found += 1
        else:
            no_face_found.append(img_path)
            
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
        
    total_processed += 1

print(f"Total images scanned: {total_processed}")
print(f"Successfully aligned and saved: {faces_found}")
print(f"Faces not detected in: {len(no_face_found)} images")