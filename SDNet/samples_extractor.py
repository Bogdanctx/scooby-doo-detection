import os
import cv2 as cv
from Parameters import Parameters
from Utilities import Utilities
from tqdm import tqdm
import uuid

root_folders = [f"{Parameters.PATH_TRAIN_INPUT}/daphne", f"{Parameters.PATH_TRAIN_INPUT}/fred", f"{Parameters.PATH_TRAIN_INPUT}/shaggy", f"{Parameters.PATH_TRAIN_INPUT}/velma"]
annotation_paths = [f"{Parameters.PATH_TRAIN_INPUT}/daphne_annotations.txt", f"{Parameters.PATH_TRAIN_INPUT}/fred_annotations.txt", f"{Parameters.PATH_TRAIN_INPUT}/shaggy_annotations.txt", f"{Parameters.PATH_TRAIN_INPUT}/velma_annotations.txt"]

os.makedirs(Parameters.PATH_NEGATIVE_SAMPLES, exist_ok=True)
os.makedirs(Parameters.PATH_POSITIVE_SAMPLES, exist_ok=True)

print(f"PPath = {os.getcwd()}")

def process_batch(images, metadata):
    # images: list of (image, image_id)
    # metadata: dict of image_id -> list of (bounding_box, character)

    WINDOW_SIZE = 90
    STRIDE = 32

    for image, image_id in tqdm(images, desc="Processing batch"):
        image_height, image_width = image.shape

        # For each face in the ground truth, extract positive samples at multiple scales
        for (gt, character) in metadata[image_id]:
            x1, y1, x2, y2 = gt
            width = x2 - x1
            height = y2 - y1

            for scale in [1.0, 1.05, 1.1, 1.15, 1.2, 1.3]:
                side = int(max(width, height) * scale)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                nx1 = center_x - side // 2
                ny1 = center_y - side // 2
                nx2 = nx1 + side
                ny2 = ny1 + side

                # Compute padding if the box goes out of image boundaries
                pad_top = max(0, -ny1)
                pad_bottom = max(0, ny2 - image_height)
                pad_left = max(0, -nx1)
                pad_right = max(0, nx2 - image_width)

                sx1 = max(0, nx1)
                sy1 = max(0, ny1)
                sx2 = min(image_width, nx2)
                sy2 = min(image_height, ny2)

                face = image[sy1:sy2, sx1:sx2]

                if face.size > 0:
                    padded_face = cv.copyMakeBorder(face, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_REPLICATE)
                    resized_face = cv.resize(padded_face, (WINDOW_SIZE, WINDOW_SIZE))
                    path_to_save = os.path.join(Parameters.PATH_POSITIVE_SAMPLES, f"{character}_{uuid.uuid4()}.png")
                    cv.imwrite(path_to_save, resized_face)

                    # Flip the face horizontally and save
                    flipped_face = cv.flip(resized_face, 1)
                    path_to_save_flipped = os.path.join(Parameters.PATH_POSITIVE_SAMPLES, f"{character}_{uuid.uuid4()}.png")
                    cv.imwrite(path_to_save_flipped, flipped_face)

        # --- NEGATIVES (Sliding Window) ---
        for y in range(0, image_height - WINDOW_SIZE + 1, STRIDE):
            for x in range(0, image_width - WINDOW_SIZE + 1, STRIDE):
                window_coordinates = (x, y, x + WINDOW_SIZE, y + WINDOW_SIZE)
                patch = image[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]

                # Check for overlap with ground truth boxes
                ious = [Utilities.intersection_over_union(window_coordinates, gt[0]) for gt in metadata[image_id]]
                max_iou = max(ious)

                if max_iou < 0.2:
                    path_to_save = os.path.join(Parameters.PATH_NEGATIVE_SAMPLES, f"{uuid.uuid4()}.png")
                    cv.imwrite(path_to_save, patch)





for root_folder, annotations_path in zip(root_folders, annotation_paths):
    
    metadata = dict()
    with open(annotations_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(' ')

        image_id = parts[0]
        x1, y1, x2, y2 = map(int, parts[1:5])
        character = parts[5]

        if image_id not in metadata:
            metadata[image_id] = []

        metadata[image_id].append([(x1, y1, x2, y2), character])


    images = os.listdir(root_folder)
    patches = [] # (image, image_id)
    for image_id in images:
        image_path = os.path.join(root_folder, image_id)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        patches.append((image, image_id))

        if len(patches) == Parameters.BATCH_SIZE:
            process_batch(patches, metadata)
            patches = []
    
    if len(patches) > 0:
        process_batch(patches, metadata)
