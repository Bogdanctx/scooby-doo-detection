import torch
from torch.utils.data import DataLoader
from Parameters import Parameters
from sklearn.metrics import classification_report

class FaceDetector(torch.nn.Module):
    def __init__(self):
        super(FaceDetector, self).__init__()

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = None


        self.training_losses = []
        self.validation_losses = []

        self.training_accuracies = []
        self.validation_accuracies = []
        
        
        self.layers = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 45x45

            # Block 2
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 22x22

            # Block 3
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 11x11

            # Block 4
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 5x5

            torch.nn.Flatten(),

            torch.nn.Linear(128 * 5 * 5, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 1)
        )

        self.to(Parameters.DEVICE)

    def forward(self, x):
        return self.layers(x)


    def fit(self, train_dataset, validation_dataset=None, learning_rate=0.001):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(Parameters.EPOCHS):
            self.train()
            print(f"Epoch {epoch+1}/{Parameters.EPOCHS}")

            correct = 0
            running_loss = 0.0
            y_true = []
            y_pred = []
            for image, label in DataLoader(train_dataset, batch_size=Parameters.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True):
                # Convert to float
                image = image.float()
                label = label.float()

                # Move to device
                image = image.to(Parameters.DEVICE)
                label = label.to(Parameters.DEVICE)

                # Forward pass
                outputs = self.forward(image)

                # Compute loss
                loss = self.criterion(outputs, label)


                # Clear gradients
                self.optimizer.zero_grad()

                # Perform backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                # Update running loss
                running_loss += loss.item() * label.size(0)
                probabilities = torch.sigmoid(outputs)

                hard_label = (label > 0.5).float() # because ground truth labels are 0.05 or 0.95
                predicted = (probabilities > 0.5).float()
                correct += (predicted == hard_label).sum().item()

                y_true.extend(hard_label.cpu().numpy().ravel())
                y_pred.extend(predicted.cpu().numpy().ravel())


            running_loss = running_loss / len(train_dataset)
            train_accuracy = correct / len(train_dataset)
            print(f"Training Loss: {running_loss:.4f} | Training Accuracy: {train_accuracy:.4f}")
        
            if validation_dataset is not None:
                val_accuracy, val_loss = self.evaluate(validation_dataset, print_report=(epoch == Parameters.EPOCHS - 1))
                print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")


    def evaluate(self, dataset, print_report=False):
        self.eval()

        correct = 0
        running_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for image, label in DataLoader(dataset, batch_size=Parameters.BATCH_SIZE, shuffle=False):
                # Convert to float
                image = image.float()
                label = label.float()

                # Move to device
                image = image.to(Parameters.DEVICE)
                label = label.to(Parameters.DEVICE)

                # Forward pass
                outputs = self.forward(image)

                # Compute loss
                loss = self.criterion(outputs, label)

                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                hard_label = (label > 0.5).float()

                correct += (predicted == hard_label).sum().item()
                running_loss += loss.item() * label.size(0)

                y_true.extend(hard_label.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = correct / len(dataset)
        running_loss = running_loss / len(dataset)

        if print_report:
            print("[INFO] Classification Report:")
            print(classification_report(y_true, y_pred, digits=4))

        return accuracy, running_loss

    
    def predict_proba(self, x):         
        self.eval()

        with torch.no_grad():
            output = self.forward(x)
            probabilities = torch.sigmoid(output)
        
        return probabilities.cpu().numpy()
    