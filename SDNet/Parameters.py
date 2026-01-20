import torch

class Parameters:
    TRAIN_DETECTION_MODEL = False
    TRAIN_RECOGNITION_MODEL = False

    PATH_TRAIN_INPUT = "./antrenare"
    PATH_TEST_INPUT = "./testare"

    PATH_POSITIVE_SAMPLES = "./positive_samples"
    PATH_NEGATIVE_SAMPLES = "./negative_samples"
    PATH_MINED_HARD_NEGATIVES = "./mined_hard_negatives"
    PATH_FACIAL_DETECTOR = "./facial_detector_model.pkl"
    PATH_FACIAL_RECOGNITION = "./facial_recognition_model.pkl"

    EPOCHS = 6
    BATCH_SIZE = 512
    WINDOW_SIZE = 90
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
