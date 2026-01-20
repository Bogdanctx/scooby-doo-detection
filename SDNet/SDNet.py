import numpy as np
import cv2 as cv
import torch
import os
from PIL import Image
from Parameters import Parameters
from Utilities import Utilities
from FaceDetector import FaceDetector
from FaceRecognition import FaceRecognition
from FaceDetectorImageDataset import FaceDetectorImageDataset
from FaceRecognitionImageDataset import FaceRecognitionImageDataset
from sklearn.model_selection import train_test_split

class SDNet:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognition(4)

        self.face_detector.load_state_dict(torch.load("./SDNet/models/face_detector.pkl", map_location=Parameters.DEVICE))
        self.face_recognizer.load_state_dict(torch.load("./SDNet/models/face_recognizer.pkl", map_location=Parameters.DEVICE))

        self.labels_map = {"Daphne": 0, "Fred": 1, "Shaggy": 2, "Velma": 3}

    def train(self):
        positives = os.listdir("./SDNet/positives/")
        negatives = os.listdir("./SDNet/negatives/")

        dataset = [(path, 0.95) for path in positives] + [(path, 0.05) for path in negatives]

        train_paths, validation_paths = train_test_split(dataset, test_size=0.3, random_state=42)

        # Detector Dataset
        self.detector_dataset = FaceDetectorImageDataset(train_paths, augment=True)
        self.detector_validation_dataset = FaceDetectorImageDataset(validation_paths)

        self.face_detector.train(self.detector_dataset, validation_dataset=self.detector_validation_dataset)

        # Recognizer Dataset
        daphne = [p for p in positives if "daphne" in p]
        fred = [p for p in positives if "fred" in p]
        shaggy = [p for p in positives if "shaggy" in p]
        velma = [p for p in positives if "velma" in p]

        dataset = [(path, self.labels_map["daphne"]) for path in daphne] + \
                  [(path, self.labels_map["fred"]) for path in fred] + \
                  [(path, self.labels_map["shaggy"]) for path in shaggy] + \
                  [(path, self.labels_map["velma"]) for path in velma]
        
        train, validation = train_test_split(dataset, test_size=0.2, random_state=42, stratify=[label for _, label in dataset])
        train_dataset = FaceRecognitionImageDataset(train, augment=True)
        validation_dataset = FaceRecognitionImageDataset(validation, augment=False)

        self.face_recognizer.train(train_dataset, validation_dataset)


        # Save models
        torch.save(self.face_detector.state_dict(), "./SDNet/models/face_detector.pkl")
        torch.save(self.face_recognizer.state_dict(), "./SDNet/models/face_recognizer.pkl")

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
        # Convert PIL to Numpy Array (RGB)
        image = np.array(img.convert("RGB"))
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        STRIDE = 12


        # apply sliding window on different scales
        scales = np.linspace(start=0.2, stop=2.0, num=65)

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

        detected_faces = []
        if len(detections) > 0:
            # Apply non-maximal suppression
            boxes_array = np.array([d[:4] for d in detections], dtype=np.float32)
            scores_array = np.array([d[4] for d in detections], dtype=np.float32)
            size = image.shape

            final_boxes, final_scores = Utilities.non_maximal_suppression(boxes_array, scores_array, size)

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

                if max_score < 0.90:
                    label_name = "Unknown"
                else:
                    label_name = [name for name, idx in self.labels_map.items() if idx == label_idx][0]

                detected_faces.append({
                    "patch": patch,
                    "label": label_name,
                    "detection_score": float(score),
                    "recognition_score": float(max_score)
                })

        result = {
            "image_with_detections": image,
            "detected_faces": detected_faces
        }

        return result
    