import json
import os
import shutil
import h5py
from tqdm import tqdm
# from data.a2d_dataset import get_masks
import numpy as np
import cv2
from imantics import Polygons, Mask
import json
# from data.a2d_dataset import generate_split_dict
import csv

def generate_split_dict(video_set_file):
    split_dict = {}
    with open(video_set_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vid, split = row[0], int(row[-1])
            split_dict[vid] = split
    return split_dict


def get_id_masks_bbox(mask_path):
    f = h5py.File(mask_path, 'r')
    masks, bboxes, ids = f['reMask'], f['reBBox'], f['id'][0]
    num_instances = len(ids)

    masks_list, bbox_list, id_list = [], [], []

    if num_instances == 1:
        mask = np.array(masks).T
        ids = str(ids[0])[0]
        x, y, w, h = bboxes[0][0], bboxes[1][0], bboxes[2][0] - bboxes[0][0], bboxes[3][0] - bboxes[1][0]
        masks_list.append(mask)
        bbox_list.append((int(x), int(y), int(w), int(h)))
        id_list.append(ids)

    else:
        for i in range(num_instances):
            mask = np.array(masks[i]).T
            ids_int = str(ids[i])[0]
            x, y, w, h = bboxes[0][i], bboxes[1][i], bboxes[2][i] - bboxes[0][i], bboxes[3][i] - bboxes[1][i]
            masks_list.append(mask)
            bbox_list.append((int(x), int(y), int(w), int(h)))
            id_list.append(ids_int)

    return masks_list, bbox_list, id_list


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    # boxes = np.zeros([1, 4], dtype=np.int32)
    # for i in range(mask.shape[-1]):
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    # boxes[i] = np.array([y1, x1, y2, x2])
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def convert_mask_to_poly(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []

    for object in contours:
        coords = []

        for point in object:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))
        if len(coords) > 4:
            polygons.append(coords)

    return polygons


# copy image to a directory
src_data_root = '../Dataset/A2D/'
frame_root = os.path.join(src_data_root, 'Release', 'Frames')
anno_root = os.path.join(src_data_root, 'Release', 'Annotations', 'mat')
video_set_path = os.path.join(src_data_root, 'Release', 'videoset.csv')

target_root = os.path.join(src_data_root, 'images')

split_dict = generate_split_dict(video_set_path)
os.makedirs(target_root, exist_ok=True)

# for seq in tqdm(os.listdir(frame_root)):
#     frame_dir = os.path.join(frame_root, seq)
#     anno_dir = os.path.join(anno_root, seq)
#     if os.path.exists(anno_dir):
#         frames_idx = [name[:-3] for name in os.listdir(anno_dir)]
#         for frame_idx in frames_idx:
#             image_path = os.path.join(frame_dir, frame_idx+'.jpg')
#             target_path = os.path.join(target_root, seq+'_'+frame_idx+'.jpg')
#             shutil.copy(image_path, target_path)


image_list_train = []
annotation_list_train = []

image_list_test = []
annotation_list_test = []
count = 0
for i, file_name in tqdm(enumerate(os.listdir(target_root))):
    image_dict = {}
    # anno_dict = {}
    # print(file_name)
    vid = file_name[:-10]
    mask_path = os.path.join(anno_root, file_name[:-10], file_name.split('_')[-1][:-4]+'.mat')
    image_path = os.path.join(target_root, file_name)
    img = cv2.imread(image_path)
    height, width = img.shape[0], img.shape[1]
    image_dict.update({"id": i, "width": width, 'height': height, "file_name": file_name})
    # print(mask_path)
    masks_list, bbox_list, ids_list = get_id_masks_bbox(mask_path)

    for j in range(len(ids_list)):
        anno_dict = {}
        segmentation = convert_mask_to_poly(masks_list[j].astype(np.uint8))
        x, y, w, h = bbox_list[j]
        anno_dict.update(
            {'segmentation': segmentation, 'area': w * h, 'iscrowd': 0, 'image_id': i, 'bbox': [x, y, w, h],
             'category_id': int(ids_list[j]), 'id': len(annotation_list_train) + len(annotation_list_test)})
        if split_dict[vid] == 0:
            annotation_list_train.append(anno_dict)
        else:
            annotation_list_test.append(anno_dict)

    # masks = get_masks(mask_path)
    # for j in range(len(masks_list)):
    #     anno_dict = {}
    #     mask = masks[j].astype(np.int)
    #     x, y, w, h = extract_bboxes(mask)
    #     # polygons = Mask(mask).polygons()
    #     # segmentation = polygons.segmentation
    #     # if len(segmentation) > 1:
    #     #     count += 1
    #     segmentation = convert_mask_to_poly(mask.astype(np.uint8))
    #     # print(type(segmentation[0]))
    #     anno_dict.update({'segmentation': segmentation, 'area': w * h, 'iscrowd': 0, 'image_id': i, 'bbox': [x, y, w, h], 'category_id': 0, 'id': len(annotation_list_train)+len(annotation_list_test)})
    #     if split_dict[vid] == 0:
    #         annotation_list_train.append(anno_dict)
    #     else:
    #         annotation_list_test.append(anno_dict)
    if split_dict[vid] == 0:
        image_list_train.append(image_dict)
    else:
        image_list_test.append(image_dict)
    # image_list.append(image_dict)
    # json_dict = {'images': image_list, 'annotations': annotation_list}

    # with open('./a2d_coco.json', 'w') as f:
    #     json.dump(json_dict, f)
    # print('sucess')
    # break
print(count)

category_list = [{'id': 1, 'name': 'adult'}, {'id': 2, 'name': 'baby'}, {'id': 3, 'name': 'ball'}, {'id': 4, 'name': 'bird'}, {'id': 5, 'name': 'car'}, {'id': 6, 'name': 'cat'}, {'id': 7, 'name': 'dog'}]

json_dict_train = {'images': image_list_train, 'annotations': annotation_list_train, 'categories': category_list}

with open('./a2d_coco_train.json', 'w') as f:
    json.dump(json_dict_train, f)

json_dict_test = {'images': image_list_test, 'annotations': annotation_list_test, 'categories': category_list}

with open('./a2d_coco_test.json', 'w') as f:
    json.dump(json_dict_test, f)




