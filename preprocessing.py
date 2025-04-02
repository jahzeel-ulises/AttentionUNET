import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import os


###############################################################
# Before running this script, you must download the dataset, #
# and rename the folder as 'brain_data'.                     #
# Also you must create a folder named numpy_brain_data, with #
# folders train, val, test. Each of these folders must have  #
# a inputs folder and outputs folder.                        #
###############################################################

def load_image_and_mask(coco, image_dir, image_id):
    image_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(image_dir, image_info['file_name'])
    image = Image.open(image_path).convert("L")
    image = np.array(image)

    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((image_info['height'], image_info['width']),dtype='uint8')
    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))

    return image, mask

def preprocess(image, mask):
    
    image = torch.from_numpy(image).unsqueeze(0).float()  # (H, W, C) → (C, H, W)
    mask = torch.from_numpy(mask).unsqueeze(0).float()  # (H, W) → (1, H, W)

    image = image / 255.0  
    
    image = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
    

    mask = F.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
    mask = (mask > 0.5).long()
    
    return image, mask

def create_dataset(coco, image_dir, image_ids,destination_dir):
    for image_id in image_ids:
        image , mask = load_image_and_mask(coco,image_dir,image_id)
        processed_image, processed_mask = preprocess(image,mask)

        np.save(destination_dir+"/inputs/"+str(image_id)+".npy",processed_image.numpy())
        np.save(destination_dir+"/outputs/"+str(image_id)+".npy",processed_mask.numpy())

#Set dataset dir
train_dir = 'brain_data/train'
val_dir = 'brain_data/valid'
test_dir = 'brain_data/test'

#Set annotations json dir
train_annotation_file = 'brain_data/train/_annotations.coco.json'
test_annotation_file = 'brain_data/test/_annotations.coco.json'
val_annotation_file = 'brain_data/valid/_annotations.coco.json'

#Defining COCO helper classes
train_coco = COCO(train_annotation_file)
val_coco = COCO(val_annotation_file)
test_coco = COCO(test_annotation_file)

#Setting 
train_dataset_dir = "numpy_brain_data/train"
test_dataset_dir = "numpy_brain_data/test"
val_dataset_dir = "numpy_brain_data/val"

create_dataset(train_coco,train_dir,train_coco.getImgIds(),train_dataset_dir)
create_dataset(test_coco,test_dir,test_coco.getImgIds(),test_dataset_dir)
create_dataset(val_coco,val_dir,val_coco.getImgIds(),val_dataset_dir)