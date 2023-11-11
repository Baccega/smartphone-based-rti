import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from constants import constants
from myIO import inputCoin
from utils import (
    generateGaussianMatrix,
    getChoosenCoinVideosPaths,
    loadDataFile,
    writeDataFile,
    fromIndexToLightDir,
    normalizeXY,
    getPytorchDevice,
)
import cv2 as cv

device = getPytorchDevice()
torch.manual_seed(42)


class ExtractedPixelsDataset(Dataset):
    def __init__(self, extracted_data_file_path, extracted_data=None):
        if extracted_data is None:
            extracted_data = loadDataFile(extracted_data_file_path)

        extracted_datapoints = list(extracted_data[0][0].keys())
        n_extracted_datapoints = len(extracted_datapoints)

        print("Number of extracted light directions: {}".format(n_extracted_datapoints))

        total = (
            constants["SQUARE_GRID_DIMENSION"]
            * constants["SQUARE_GRID_DIMENSION"]
            * n_extracted_datapoints
        )
        # Length = size of 2 positions + 2 light directions + label
        self.data = torch.empty([total, 2 + 2 + 1], dtype=torch.float32)
        # self.data = np.empty([total, 2 + 2 + 1])

        print("Preparing dataset data")
        for x in tqdm(range(constants["SQUARE_GRID_DIMENSION"])):
            normalized_x = normalizeXY(x)
            for y in range(constants["SQUARE_GRID_DIMENSION"]):
                normalized_y = normalizeXY(y)
                for z in range(n_extracted_datapoints):
                    lightDirection = extracted_datapoints[z]
                    light_direction_x = float(lightDirection.split("|")[0])
                    light_direction_y = float(lightDirection.split("|")[1])
                    i = (
                        (
                            x
                            * constants["SQUARE_GRID_DIMENSION"]
                            * n_extracted_datapoints
                        )
                        + (y * n_extracted_datapoints)
                        + z
                    )
                    self.data[i] = torch.cat(
                        (
                            torch.tensor(
                                [
                                    normalized_x,
                                    normalized_y,
                                    light_direction_x,
                                    light_direction_y,
                                ]
                            ),
                            torch.tensor([extracted_data[x][y][lightDirection]]),
                        ),
                        dim=-1,
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx][:-1]

        # Add noise to light direction in input
        noise = np.random.normal(loc=0, scale=0.005)
        # input[2:] += noise
    
        label = self.data[idx][-1]
        return input, label


class NeuralModel(nn.Module):
    def __init__(self, gaussian_matrix_xy, gaussian_matrix_uv):
        super(NeuralModel, self).__init__()

        self.register_buffer(
            "gaussian_matrix_xy",
            torch.tensor(gaussian_matrix_xy.astype(np.float32)),
            persistent=True,
        )
        self.register_buffer(
            "gaussian_matrix_uv",
            torch.tensor(gaussian_matrix_uv.astype(np.float32)),
            persistent=True,
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(constants["NEURAL_INPUT_SIZE"], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        position = (6.283185 * (x[:, :2] @ self.gaussian_matrix_xy)).clone().detach()
        position = torch.cat([torch.cos(position), torch.sin(position)], dim=-1)

        light = (6.283185 * (x[:, -2:] @ self.gaussian_matrix_uv)).clone().detach()
        light = torch.cat([torch.cos(light), torch.sin(light)], dim=-1)
        x = torch.cat([position, light], dim=-1)
        out = self.linear_relu_stack(x)
        return out


def train_neural_model(
    model_path,
    extracted_data,
    gaussian_matrix_xy,
    gaussian_matrix_uv,
    extracted_data_file_path="Already extracted",
):
    print("Neural model: " + model_path)
    print("Training data: " + extracted_data_file_path)

    model = NeuralModel(
        gaussian_matrix_xy=gaussian_matrix_xy, gaussian_matrix_uv=gaussian_matrix_uv
    )
    model = model.to(device)

    dataset = ExtractedPixelsDataset(
        extracted_data_file_path, extracted_data=extracted_data
    )
    dataloader = DataLoader(
        dataset, batch_size=constants["NEURAL_BATCH_SIZE"], shuffle=True, num_workers=4
    )

    # Mean Absolute Error
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=constants["NEURAL_LEARNING_RATE"])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    print("Starting training:")

    for epoch in range(
        constants["NEURAL_N_EPOCHS"]
    ):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

                running_loss += loss.item()
                loss.backward()
                optimizer.step()

        scheduler.step()
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        loss = running_loss / (
            constants["SQUARE_GRID_DIMENSION"] * constants["SQUARE_GRID_DIMENSION"]
        )
        print(f"Epoch {epoch + 1}, L1 loss: {loss}, lr: {current_lr}")

    print("Finished Training")

    torch.save(model.state_dict(), model_path)

    print("Saved model to {}".format(model_path))


def main():
    print("Neural model training")

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
        _,
        _,
    ) = getChoosenCoinVideosPaths(coin)

    if not os.path.exists(extracted_data_file_path):
        raise (
            Exception(
                "You need to extract the coin data with the analysis before training the model!"
            )
        )

    # gaussian_matrix = np.random.randn(2, 10) * sigma
    if not os.path.exists(
        constants["GAUSSIAN_MATRIX_FILE_PATH_XY"]
    ) or not os.path.exists(constants["GAUSSIAN_MATRIX_FILE_PATH_UV"]):
        gaussian_matrix_xy = generateGaussianMatrix(
            0, torch.tensor(constants["NEURAL_SIGMA_XY"]), constants["NEURAL_H_XY"]
        )
        gaussian_matrix_uv = generateGaussianMatrix(
            0, torch.tensor(constants["NEURAL_SIGMA_UV"]), constants["NEURAL_H_UV"]
        )
        writeDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"], gaussian_matrix_xy)
        writeDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"], gaussian_matrix_uv)
    else:
        gaussian_matrix_xy = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH_XY"])
        gaussian_matrix_uv = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH_UV"])

    train_neural_model(
        model_path,
        None,
        gaussian_matrix_xy,
        gaussian_matrix_uv,
        extracted_data_file_path,
    )


if __name__ == "__main__":
    main()
