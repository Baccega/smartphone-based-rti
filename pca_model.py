import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from constants import constants
from myIO import inputCoin
from utils import (
    generateGaussianMatrix,
    getProjectedLightsInFourierSpace,
    getChoosenCoinVideosPaths,
    loadDataFile,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# Hyperparameters

N_CLASSES = 102
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
N_EPOCHS = 40

# Local constants

B = 8
H = 10
sigma = 0.3

MODEL_INPUT_SIZE = B + (2 * H)


class PixelDataset(Dataset):
    def __init__(self, extracted_data_file_path, B):
        self.B = B
        extracted_data = loadDataFile(extracted_data_file_path)

        total = (
            constants["SQAURE_GRID_DIMENSION"]
            * constants["SQAURE_GRID_DIMENSION"]
            * len(list(extracted_data[0][0].keys()))
        )
        self.data = np.empty([total, 3])

        for x in tqdm(range(constants["SQAURE_GRID_DIMENSION"])):
            for y in range(constants["SQAURE_GRID_DIMENSION"]):
                keys = list(extracted_data[x][y].keys())
                light_directions_x = [i.split("|")[0] for i in keys]
                light_directions_y = [i.split("|")[1] for i in keys]
                pixel_intensities = list(extracted_data[x][y].values())
                for i in range(len(pixel_intensities)):
                    j = x * y * i
                    self.data[j][0] = light_directions_x[i]
                    self.data[j][1] = light_directions_y[i]
                    self.data[j][2] = pixel_intensities[i]

    def __len__(self):
        # total = (
        #     constants["SQAURE_GRID_DIMENSION"]
        #     * constants["SQAURE_GRID_DIMENSION"]
        #     * len(list(self.data[0][0].keys()))
        # )
        return len(self.data)

    def __getitem__(self, idx):
        light_direction_x, light_direction_y = (self.data[idx][0], self.data[idx][1])
        cos_s, sin_s = getProjectedLightsInFourierSpace(light_direction_x, light_direction_y, self.B)

        # print("COS: ", np.cos(s))
        # print("SIN: ", np.sin(s))

        # input = pca + cos(s) + sin(s) 

        input = torch.cat((cos_s, sin_s), 0)
        label = torch.tensor(self.data[idx][2])
        return input, label


class NeuralModel(nn.Module):
    def __init__(
        self,
    ):
        super(NeuralModel, self).__init__()
        self.fc1 = nn.Linear(MODEL_INPUT_SIZE, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = F.elu(self.fc5(x))
        return x


def train(model_path, extracted_data_file_path, B):
    print("PCA model: " + model_path)
    print("Training data: " + extracted_data_file_path)

    model = NeuralModel()

    dataset = PixelDataset(extracted_data_file_path, B)

    # Random split (90% - 10%)
    train_set_size = int(len(dataset) * 0.9)
    test_set_size = len(dataset) - train_set_size
    training_dataset, test_dataset = random_split(
        dataset, [train_set_size, test_set_size]
    )

    train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    def check_accuracy(data_loader, model):
        n_corrects = 0
        n_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in data_loader:
                # Sending data to device
                x = x.to(device)
                y = y.to(device)

                # Forward propagation
                y_hat = model(x)

                # Calculate accuracy
                _, predictions = y_hat.max(1)
                n_corrects += (predictions == y).sum()
                n_samples += predictions.size(0)

            perc = (n_corrects.item() / n_samples) * 100
            return (n_corrects.item(), n_samples, perc)

    # Mean Absolute Error
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # training_results_data = pd.DataFrame(
    #     {"Epoch": [], "Predictions": [], "Samples": [], "Accuracy": [], "Loss": []}
    # )

    for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            scheduler.step()
            current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
            n_corrects, n_samples, accuracy = check_accuracy(test_dataloader, model)
            # training_results_data.loc[len(training_results_data.index)] = [
            #     int(epoch + 1),
            #     int(n_corrects),
            #     int(n_samples),
            #     accuracy,
            #     running_loss,
            # ]
            print(f"Epoch {epoch + 1}, loss: {running_loss}, lr: {current_lr}")
            print(f"Accuracy: {n_corrects}/{n_samples} = {accuracy:.2f}%")

    print("Finished Training")

    torch.save(model.state_dict(), model_path)


def main():
    print("PCA model training")

    coin = inputCoin()
    (
        _,
        _,
        _,
        _,
        _,
        extracted_data_file_path,
        _,
        model_path,
    ) = getChoosenCoinVideosPaths(coin)

    if not os.path.exists(extracted_data_file_path):
        raise (
            Exception(
                "You need to extract the coin data with the analysis before training the model!"
            )
        )

    B = generateGaussianMatrix(0, torch.tensor(sigma), H)
    # B = generateGaussianMatrix(0, torch.tensor(sigma), H * 2)

    train(model_path, extracted_data_file_path, B)

if __name__ == "__main__":
    main()
