import numpy as np
import cv2 as cv
import torch
import os
import sys
import subprocess
from PIL import Image
from Parameters import Parameters
from Utilities import Utilities
from FaceDetector import FaceDetector
from FaceRecognition import FaceRecognition
from FaceDetectorImageDataset import FaceDetectorImageDataset
from FaceRecognitionImageDataset import FaceRecognitionImageDataset
from sklearn.model_selection import train_test_split
from HardNegatives import HardNegativeMiner
from Logger import Logger

class SDNet:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognition(5)

        self.labels_map = {"daphne": 0, "fred": 1, "shaggy": 2, "velma": 3, "unknown": 4}

    def load_models(self):
        self.face_detector.load_state_dict(torch.load(Parameters.PATH_FACIAL_DETECTOR, map_location=Parameters.DEVICE))
        self.face_recognizer.load_state_dict(torch.load(Parameters.PATH_FACIAL_RECOGNITION, map_location=Parameters.DEVICE))

    
    def train_detector(self):
        Logger.log("Preparing data for Face Detector training...")
        positives = [os.path.join(f"{Parameters.PATH_POSITIVE_SAMPLES}/", p) for p in os.listdir(Parameters.PATH_POSITIVE_SAMPLES)]
        negatives = [os.path.join(Parameters.PATH_NEGATIVE_SAMPLES, n) for n in os.listdir(Parameters.PATH_NEGATIVE_SAMPLES)]

        Logger.log(f"Training Face Detector with {len(positives)} positives and {len(negatives)} negatives.")

        dataset = [(path, 0.95) for path in positives] + [(path, 0.05) for path in negatives]

        train_paths, validation_paths = train_test_split(dataset, test_size=0.3, random_state=42)

        ### Detector Dataset ###
        self.detector_dataset = FaceDetectorImageDataset(train_paths, augment=True)
        self.detector_validation_dataset = FaceDetectorImageDataset(validation_paths)

        Logger.log("[INFO] Starting Face Detector training loop (fit)...")
        self.face_detector.fit(train_dataset=self.detector_dataset,
                                 validation_dataset=self.detector_validation_dataset)
        Logger.log("[INFO] Face Detector training completed.")
        #############

        # Save the trained detector model
        torch.save(self.face_detector.state_dict(), f"{Parameters.PATH_FACIAL_DETECTOR}")
        Logger.log(f"[INFO] Face Detector model saved to {Parameters.PATH_FACIAL_DETECTOR}.")

        Logger.log("[INFO] Reloading models...")
        self.load_models()
        Logger.log("[INFO] Detector training pipeline finished successfully.")

    def train_recognizer(self):
        Logger.log("[INFO] Preparing data for Face Recognizer training...")
        positives = [os.path.join(Parameters.PATH_POSITIVE_SAMPLES, p) for p in os.listdir(Parameters.PATH_POSITIVE_SAMPLES)]

        Logger.log(f"[INFO] Training Face Recognizer with {len(positives)} faces.")

        ### Recognizer Dataset ###
        daphne = [p for p in positives if "daphne" in p]
        fred = [p for p in positives if "fred" in p]
        shaggy = [p for p in positives if "shaggy" in p]
        velma = [p for p in positives if "velma" in p]
        unknown = [p for p in positives if "unknown" in p]

        Logger.log(f"[INFO] Dataset distribution: Daphne: {len(daphne)}, Fred: {len(fred)}, Shaggy: {len(shaggy)}, Velma: {len(velma)}, Unknown: {len(unknown)}")

        dataset = [(path, self.labels_map["daphne"]) for path in daphne] + \
                [(path, self.labels_map["fred"]) for path in fred] + \
                [(path, self.labels_map["shaggy"]) for path in shaggy] + \
                [(path, self.labels_map["velma"]) for path in velma] + \
                [(path, self.labels_map["unknown"]) for path in unknown]
        
        train, validation = train_test_split(dataset, test_size=0.2, random_state=42, stratify=[label for _, label in dataset])
        train_dataset = FaceRecognitionImageDataset(train, augment=True)
        validation_dataset = FaceRecognitionImageDataset(validation, augment=False)

        Logger.log("[INFO] Starting Face Recognizer training loop (fit)...")
        self.face_recognizer.fit(train_dataset=train_dataset, validation_dataset=validation_dataset)
        Logger.log("[INFO] Face Recognizer training completed.")
        ###############

        # Save the trained recognizer model
        torch.save(self.face_recognizer.state_dict(), f"{Parameters.PATH_FACIAL_RECOGNITION}")
        Logger.log(f"[INFO] Face Recognizer model saved to {Parameters.PATH_FACIAL_RECOGNITION}.")

        Logger.log("[INFO] Reloading models...")
        self.load_models()
        Logger.log("[INFO] Recognizer training pipeline finished successfully.")

    def train(self):
        Logger.log("[INFO] Starting full training pipeline...")
        # Move collected samples to training folders
        Logger.log("[INFO] Checking for collected samples...")
        Utilities.move_collected_samples()

        if len(os.listdir(f"{Parameters.PATH_POSITIVE_SAMPLES}")) == 0 or len(os.listdir(f"{Parameters.PATH_NEGATIVE_SAMPLES}")) == 0:
            Logger.log("[INFO] No samples found in positives/negatives. Running samples_extractor.py...")
            subprocess.run(
                [sys.executable, "-u", "./SDNet/samples_extractor.py"],
                cwd=os.getcwd(),
                check=True
            )

        # Train initial detector
        Logger.log("[INFO] Step 1: Training initial detector...")
        self.train_detector()

        # Mine hard negatives
        Logger.log("[INFO] Step 2: Mining hard negatives...")
        miner = HardNegativeMiner(self.face_detector)
        miner.mine_hard_negatives()

        # Retrain detector with new hard negatives
        Logger.log("[INFO] Step 3: Retraining detector with hard negatives...")
        self.train_detector()

        # Train recognizer
        Logger.log("[INFO] Step 4: Training recognizer...")
        self.train_recognizer()

    def evaluate(self):
        validation_folder = f"{Parameters.PATH_TEST_INPUT}"
        val_files = [os.path.join(validation_folder, f) for f in os.listdir(validation_folder)]

        all_detections = []
        all_scores = []
        all_filenames = []

        detections_daphne = []
        detections_fred = []
        detections_shaggy = []
        detections_velma = []

        scores_daphne = []
        scores_fred = []
        scores_shaggy = []
        scores_velma = []

        file_names_daphne = []
        file_names_fred = []
        file_names_shaggy = []
        file_names_velma = []

        print("[INFO] Starting validation")
        STRIDE = 4


        for index, image_path in enumerate(val_files):
            image_name = image_path.split("/")[-1]

            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            detections = []

            for scale in np.linspace(start=0.2, stop=2.0, num=65):
                scaled_image = cv.resize(image, None, fx=scale, fy=scale)
                image_height, image_width = scaled_image.shape
                patches = []
                metadata = []

                for y in range(0, image_height - Parameters.WINDOW_SIZE + 1, STRIDE):
                    for x in range(0, image_width - Parameters.WINDOW_SIZE + 1, STRIDE):
                        window = scaled_image[y:y+Parameters.WINDOW_SIZE, x:x+Parameters.WINDOW_SIZE]
                        patches.append(window)
                        metadata.append((x, y, scale))

                        if len(patches) == Parameters.BATCH_SIZE:
                            batch_detections = self.batch_detect(patches, metadata)
                            detections.extend(batch_detections)
                            patches = []
                            metadata = []

                # Process remaining patches
                if len(patches) > 0:
                    batch_detections = self.batch_detect(patches, metadata)
                    detections.extend(batch_detections)
                    patches = []
                    metadata = []

            if len(detections) > 0:
                # Apply non-maximal suppression
                boxes_array = np.array([d[:4] for d in detections], dtype=np.float32)
                scores_array = np.array([d[4] for d in detections], dtype=np.float32)
                size = image.shape

                final_boxes, final_scores = Utilities.non_maximal_suppression(
                    boxes_array, scores_array, size
                )

                for i in range(len(final_boxes)):
                    all_detections.append(final_boxes[i])
                    all_scores.append(final_scores[i])
                    all_filenames.append(image_name)

                for (x_min, y_min, x_max, y_max), score in zip(final_boxes, final_scores):
                    x_min = max(0, int(x_min))
                    y_min = max(0, int(y_min))
                    x_max = min(image.shape[1], int(x_max))
                    y_max = min(image.shape[0], int(y_max))
                    
                    patch = image[y_min:y_max, x_min:x_max]
                    patch_resized = cv.resize(patch, (Parameters.WINDOW_SIZE, Parameters.WINDOW_SIZE))
                    patch_tensor = torch.tensor(patch_resized / 255.0 - 0.5, dtype=torch.float32)
                    patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)

                    label = self.face_recognizer.predict(patch_tensor)
                    label_name = [name for name, idx in self.labels_map.items() if idx == label][0]

                    if label_name == "daphne":
                        detections_daphne.append((x_min, y_min, x_max, y_max))
                        scores_daphne.append(score)
                        file_names_daphne.append(image_name)
                    elif label_name == "fred":
                        detections_fred.append((x_min, y_min, x_max, y_max))
                        scores_fred.append(score)
                        file_names_fred.append(image_name)
                    elif label_name == "shaggy":
                        detections_shaggy.append((x_min, y_min, x_max, y_max))
                        scores_shaggy.append(score)
                        file_names_shaggy.append(image_name)
                    elif label_name == "velma":
                        detections_velma.append((x_min, y_min, x_max, y_max))
                        scores_velma.append(score)
                        file_names_velma.append(image_name)

            print(f"[INFO] Processed image {image_name} ({index + 1}/{len(val_files)})")

        print("[INFO] Validation completed")
        all_detections = np.array(all_detections)
        all_scores = np.array(all_scores)
        all_filenames = np.array(all_filenames)

        face_detector_results = f"{Parameters.PATH_TEST_INPUT}/face_detector_results/"
        face_recognizer_results = f"{Parameters.PATH_TEST_INPUT}/face_recognizer_results/"

        os.makedirs(face_detector_results, exist_ok=True)
        os.makedirs(face_recognizer_results, exist_ok=True)

        np.save(os.path.join(face_detector_results, "detections_all_faces.npy"), all_detections)
        np.save(os.path.join(face_detector_results, "scores_all_faces.npy"), all_scores)
        np.save(os.path.join(face_detector_results, "file_names_all_faces.npy"), all_filenames)

        np.save(os.path.join(face_recognizer_results, "detections_daphne.npy"), np.array(detections_daphne))
        np.save(os.path.join(face_recognizer_results, "scores_daphne.npy"), np.array(scores_daphne))
        np.save(os.path.join(face_recognizer_results, "file_names_daphne.npy"), np.array(file_names_daphne))

        np.save(os.path.join(face_recognizer_results, "detections_fred.npy"), np.array(detections_fred))
        np.save(os.path.join(face_recognizer_results, "scores_fred.npy"), np.array(scores_fred))
        np.save(os.path.join(face_recognizer_results, "file_names_fred.npy"), np.array(file_names_fred))

        np.save(os.path.join(face_recognizer_results, "detections_shaggy.npy"), np.array(detections_shaggy))
        np.save(os.path.join(face_recognizer_results, "scores_shaggy.npy"), np.array(scores_shaggy))
        np.save(os.path.join(face_recognizer_results, "file_names_shaggy.npy"), np.array(file_names_shaggy))

        np.save(os.path.join(face_recognizer_results, "detections_velma.npy"), np.array(detections_velma))
        np.save(os.path.join(face_recognizer_results, "scores_velma.npy"), np.array(scores_velma))
        np.save(os.path.join(face_recognizer_results, "file_names_velma.npy"), np.array(file_names_velma))


    def batch_detect(self, patches, metadata):
        patches = [p / 255.0 if p.max() > 1.0 else p for p in patches]
        tensor_patches = torch.tensor(np.array(patches), dtype=torch.float32, device=Parameters.DEVICE).unsqueeze(1)
        probabilities = self.face_detector.predict_proba(tensor_patches)
        probabilities = probabilities.flatten()
        detections = []

        for i, prob in enumerate(probabilities):

            if prob >= 0.90:
                x, y, scale = metadata[i][:3]
                
                x_min = int(x / scale)
                y_min = int(y / scale)
                x_max = int((x + Parameters.WINDOW_SIZE) / scale)
                y_max = int((y + Parameters.WINDOW_SIZE) / scale)

                detections.append((x_min, y_min, x_max, y_max, prob))

        return detections


    def detect_faces(self, img: Image.Image) -> dict:
        print("[INFO] Starting face detection on image...")
        # Convert PIL to Numpy Array (RGB)
        image = np.array(img.convert("RGB"))
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        print(f"[INFO] Image converted to grayscale. Size: {gray_image.shape}")
        STRIDE = 12


        # apply sliding window on different scales
        scales = np.linspace(start=0.2, stop=2.0, num=65)

        print(f"[INFO] Scanning image with {len(scales)} scales using sliding window...")
        detections = []
        for scale in scales:
            scaled_image = cv.resize(gray_image, None, fx=scale, fy=scale)
            image_height, image_width = scaled_image.shape
            patches = []
            metadata = []

            for y in range(0, image_height - Parameters.WINDOW_SIZE + 1, STRIDE):
                for x in range(0, image_width - Parameters.WINDOW_SIZE + 1, STRIDE):
                    window = scaled_image[y:y+Parameters.WINDOW_SIZE, x:x+Parameters.WINDOW_SIZE]
                    patches.append(window)
                    metadata.append((x, y, scale))

                    if len(patches) == Parameters.BATCH_SIZE:
                        batch_detections = self.batch_detect(patches, metadata)

                        detections.extend(batch_detections)
                        patches = []
                        metadata = []

            # Process remaining patches
            if len(patches) > 0:
                batch_detections = self.batch_detect(patches, metadata)
                detections.extend(batch_detections)
                patches = []
                metadata = []

        print(f"[INFO] Found {len(detections)} raw detections. Applying non-maximal suppression...")
        detected_faces = []
        if len(detections) > 0:
            # Apply non-maximal suppression
            boxes_array = np.array([d[:4] for d in detections], dtype=np.float32)
            scores_array = np.array([d[4] for d in detections], dtype=np.float32)
            size = image.shape

            final_boxes, final_scores = Utilities.non_maximal_suppression(boxes_array, scores_array, size)
            print(f"[INFO] {len(final_boxes)} faces remained after NMS. Identifying characters...")

            for (x_min, y_min, x_max, y_max), score in zip(final_boxes, final_scores):
                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(image.shape[1], int(x_max))
                y_max = min(image.shape[0], int(y_max))

                # save face patch
                patch = image[y_min:y_max, x_min:x_max].copy()

                # draw rectangle around the detected face
                cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                gray_patch = gray_image[y_min:y_max, x_min:x_max].copy()
                patch_resized = cv.resize(gray_patch, (Parameters.WINDOW_SIZE, Parameters.WINDOW_SIZE))
                patch_tensor = torch.tensor(patch_resized / 255.0 - 0.5, dtype=torch.float32)
                patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)

                probs_vector = self.face_recognizer.predict_proba(patch_tensor)
                max_score = probs_vector.max().item()
                label_idx = np.argmax(probs_vector)
                
                label_name = [name for name, idx in self.labels_map.items() if idx == label_idx][0]

                # capitalize the first letter of the label name
                label_name = label_name.capitalize()

                detected_faces.append({
                    "patch": patch,
                    "label": label_name,
                    "detection_score": float(score),
                    "recognition_score": float(max_score)
                })

        print(f"[INFO] Detection finished. Returning {len(detected_faces)} detected faces.")
        result = {
            "image_with_detections": image,
            "detected_faces": detected_faces
        }

        return result
    