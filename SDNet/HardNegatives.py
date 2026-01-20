import os
from Parameters import Parameters
from tqdm import tqdm
import cv2 as cv
import torch
import numpy as np
import uuid
from Utilities import Utilities


class HardNegativeMiner:
    def __init__(self, facial_detector):
        self.facial_detector = facial_detector

    def process_batch(self, patches, metadata):
        patches = [p / 255.0 if p.max() > 1.0 else p for p in patches]
        tensor_patches = torch.tensor(np.array(patches), dtype=torch.float32, device=Parameters.DEVICE).unsqueeze(1)
        probabilities = self.facial_detector.predict_proba(tensor_patches)
        probabilities = probabilities.flatten()

        for i, prob in enumerate(probabilities):

            if 0.5 <= prob < 0.90:
                x, y, scale = metadata[i][:3]
                
                x_min = int(x / scale)
                y_min = int(y / scale)
                x_max = int((x + Parameters.WINDOW_SIZE) / scale)
                y_max = int((y + Parameters.WINDOW_SIZE) / scale)

                gt_boxes = metadata[i][3]
                is_hard_negative = True

                for (bx1, by1, bx2, by2) in gt_boxes:
                    iou = Utilities.intersection_over_union((x_min, y_min, x_max, y_max), (bx1, by1, bx2, by2))
                    if iou > 0.3:
                        is_hard_negative = False
                        break

                if is_hard_negative:
                    patch = (patches[i] * 255).astype(np.uint8)
                    path_to_save = f"{Parameters.PATH_NEGATIVE_SAMPLES}/{uuid.uuid4()}.png"
                    cv.imwrite(path_to_save, patch)

    def mine_hard_negatives(self):
        stride = 14

        # Load annotations
        boxes = dict()
        daphne = (f"{Parameters.PATH_TRAIN_INPUT}/daphne", f"{Parameters.PATH_TRAIN_INPUT}/daphne_annotations.txt")
        fred = (f"{Parameters.PATH_TRAIN_INPUT}/fred", f"{Parameters.PATH_TRAIN_INPUT}/fred_annotations.txt")
        shaggy = (f"{Parameters.PATH_TRAIN_INPUT}/shaggy", f"{Parameters.PATH_TRAIN_INPUT}/shaggy_annotations.txt")
        velma = (f"{Parameters.PATH_TRAIN_INPUT}/velma", f"{Parameters.PATH_TRAIN_INPUT}/velma_annotations.txt")

        before_mine = len(os.listdir(Parameters.PATH_NEGATIVE_SAMPLES))
        print(f"[INFO] Starting with {before_mine} negative samples.")

        for root_path, annotations_path in [daphne, fred, shaggy, velma]:
            with open(annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    image_name = parts[0]
                    x1, y1, x2, y2 = map(int, parts[1:5])

                    image_name = parts[0]
                    image_path = os.path.join(root_path, image_name) # "./antrenare/daphne/0123.jpg"
                    x1, y1, x2, y2 = map(int, parts[1:5])

                    if image_path not in boxes:
                        boxes[image_path] = []

                    boxes[image_path].append((x1, y1, x2, y2))



        for root_path in [f"{Parameters.PATH_TRAIN_INPUT}/daphne", f"{Parameters.PATH_TRAIN_INPUT}/fred", f"{Parameters.PATH_TRAIN_INPUT}/shaggy", f"{Parameters.PATH_TRAIN_INPUT}/velma"]:
            root_images = os.listdir(root_path)

            for image in tqdm(root_images, desc=f"Mining in {root_path}"):
                image_full_path = os.path.join(root_path, image)
                original_image = cv.imread(image_full_path, cv.IMREAD_GRAYSCALE)
                patches = []
                metadata = []

                for scale in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
                    scaled_img = cv.resize(original_image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
                    height, width = scaled_img.shape

                    for y in range(0, height - Parameters.WINDOW_SIZE + 1, stride):
                        for x in range(0, width - Parameters.WINDOW_SIZE + 1, stride):
                            patch = scaled_img[y:y+Parameters.WINDOW_SIZE, x:x+Parameters.WINDOW_SIZE]
                            patches.append(patch)
                            metadata.append((x, y, scale, boxes.get(image_full_path, [])))

                            if len(patches) == 512:
                                self.process_batch(patches, metadata)
                                patches = []
                                metadata = []

                # Process remaining patches
                if len(patches) > 0:
                    self.process_batch(patches, metadata)
                    patches = []
                    metadata = []

        after_mine = len(os.listdir(Parameters.PATH_NEGATIVE_SAMPLES))
        print(f"[INFO] End with {after_mine} negative samples.")
        print(f"[INFO] Mined {after_mine - before_mine} hard negatives.")