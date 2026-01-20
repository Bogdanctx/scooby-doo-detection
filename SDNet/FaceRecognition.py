import torch
from Parameters import Parameters
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


class FaceRecognition(torch.nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognition, self).__init__()

        self.weights = torch.tensor([1.198, 1.082, 1.077, 0.897]).to(Parameters.DEVICE)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weights)

        self.optimizer = None

        self.training_losses = []
        self.validation_losses = []

        self.training_accuracies = []
        self.validation_accuracies = []
        
        
        self.layers = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 45x45

            # Block 2
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 22x22

            # Block 3
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 11x11

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),

            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

        self.to(Parameters.DEVICE)

    def forward(self, x):
        return self.layers(x)


    def fit(self, train_dataset, validation_dataset=None, learning_rate=0.001):
        self.training_accuracies = []
        self.validation_accuracies = []
        self.training_losses = []
        self.validation_losses = []

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        EPOCHS = 30
        for epoch in range(EPOCHS):
            self.train()

            print(f"Starting epoch {epoch+1}/{EPOCHS}")

            running_loss = 0.0
            correct = 0
            total = 0

            train_loader = DataLoader(train_dataset, batch_size=Parameters.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
            for images, labels in train_loader:
                images = images.to(Parameters.DEVICE)
                labels = labels.to(Parameters.DEVICE).squeeze(1).long()

                self.optimizer.zero_grad()

                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = correct / total

            self.training_losses.append(epoch_loss)
            self.training_accuracies.append(epoch_accuracy)

            print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

            if validation_dataset is not None:
                val_accuracy, val_loss = self.evaluate(validation_dataset, print_report=(epoch == EPOCHS - 1))
                self.validation_accuracies.append(val_accuracy)
                self.validation_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


    def evaluate(self, dataset, print_report=False):
        self.eval()

        correct = 0
        running_loss = 0.0
        y_true = []
        y_pred = []
        data_loader = DataLoader(dataset, batch_size=Parameters.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        with torch.no_grad():
            for image, label in data_loader:
                # Convert to float
                image = image.float()
                label = label.float()

                # Move to device
                image = image.to(Parameters.DEVICE)
                label = label.to(Parameters.DEVICE).squeeze().long()

                # Forward pass
                outputs = self.forward(image)

                # Compute loss
                loss = self.criterion(outputs, label)

                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == label).sum().item()
                running_loss += loss.item() * label.size(0)

                y_true.extend(label.cpu().numpy().ravel())
                y_pred.extend(predicted.cpu().numpy().ravel())

        total_samples = len(dataset)
        accuracy = correct / total_samples
        avg_loss = running_loss / total_samples

        if print_report:
            print(classification_report(y_true, y_pred, digits=4))

        return accuracy, avg_loss
    
    def predict_proba(self, batch):
        self.eval()
        with torch.no_grad():
            batch = batch.to(Parameters.DEVICE)
            outputs = self.forward(batch)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
    
    def predict(self, batch):
        probs = self.predict_proba(batch)
        return probs.argmax(axis=1)
    
