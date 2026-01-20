import torch

class Parameters:
    TRAIN_DETECTION_MODEL = False
    TRAIN_RECOGNITION_MODEL = False

    PATH_TRAIN_INPUT = "./SDNet/train"
    PATH_TEST_INPUT = "./SDNet/evaluation"

    PATH_POSITIVE_SAMPLES = "./SDNet/positives"
    PATH_NEGATIVE_SAMPLES = "./SDNet/negatives"
    PATH_FACIAL_DETECTOR = "./SDNet/models/face_detector.pkl"
    PATH_FACIAL_RECOGNITION = "./SDNet/models/face_recognizer.pkl"

    EPOCHS = 6
    BATCH_SIZE = 512
    WINDOW_SIZE = 90
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
