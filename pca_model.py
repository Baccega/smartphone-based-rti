import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from constants import constants
from myIO import inputCoin
from utils import (
    generateGaussianMatrix,
    getChoosenCoinVideosPaths,
    loadDataFile,
    writeDataFile,
    fromIndexToLightDir
)
import cv2 as cv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


class ExtractedPixelsDataset(Dataset):
    def __init__(self, extracted_data_file_path, pca_data_file_path):
        extracted_data = loadDataFile(extracted_data_file_path)

        n_extracted_datapoints = len(list(extracted_data[0][0].keys()))

        print("Number of extracted light directions: {}".format(n_extracted_datapoints))

        total = (
            constants["SQAURE_GRID_DIMENSION"]
            * constants["SQAURE_GRID_DIMENSION"]
            * n_extracted_datapoints
        )
        # Length = size of K + 2 light directions
        self.data = np.empty([total, 2 + constants["PCA_ORTHOGONAL_BASES"] + 1])
        full_pca_data = np.empty(
            [
                constants["SQAURE_GRID_DIMENSION"] * constants["SQAURE_GRID_DIMENSION"],
                n_extracted_datapoints,
            ]
        )

        print("Loading PCA Data")
        for i in tqdm(
            range(
                constants["SQAURE_GRID_DIMENSION"] * constants["SQAURE_GRID_DIMENSION"]
            )
        ):
            x = i % constants["SQAURE_GRID_DIMENSION"]
            y = math.floor(i / constants["SQAURE_GRID_DIMENSION"])
            full_pca_data[i] = list(extracted_data[x][y].values())

        print("Running PCA")
        pca = PCA(n_components=constants["PCA_ORTHOGONAL_BASES"])
        pca_data = pca.fit_transform(full_pca_data)

        pca_data = torch.reshape(
            torch.tensor(pca_data),
            (
                constants["SQAURE_GRID_DIMENSION"],
                constants["SQAURE_GRID_DIMENSION"],
                constants["PCA_ORTHOGONAL_BASES"],
            ),
        )

        writeDataFile(pca_data_file_path, pca_data)

        print("Preparing dataset data")

        keys = list(extracted_data[0][0].keys())
        light_directions_x = [i.split("|")[0] for i in keys]
        light_directions_y = [i.split("|")[1] for i in keys]
        for x in tqdm(range(constants["SQAURE_GRID_DIMENSION"])):
            for y in range(constants["SQAURE_GRID_DIMENSION"]):
                pixel_intensities = list(extracted_data[x][y].values())
                for z in range(n_extracted_datapoints):
                    i = (x * constants["SQAURE_GRID_DIMENSION"] * n_extracted_datapoints) + (y * n_extracted_datapoints) + z 
                    self.data[i] = torch.cat(
                        (
                            pca_data[x][y],
                            torch.tensor(
                                [
                                    float(fromIndexToLightDir(light_directions_x[z])),
                                    float(fromIndexToLightDir(light_directions_y[z])),
                                ]
                            ),
                            torch.tensor([pixel_intensities[z]]),
                        ),
                        dim=-1,
                    )


        # for i in tqdm(
        #     range(
        #         constants["SQAURE_GRID_DIMENSION"]
        #         * constants["SQAURE_GRID_DIMENSION"]
        #         * n_extracted_datapoints
        #     )
        # ):
        #     x = i % constants["SQAURE_GRID_DIMENSION"]
        #     y = (
        #         math.floor(i / constants["SQAURE_GRID_DIMENSION"])
        #         % constants["SQAURE_GRID_DIMENSION"]
        #     )
        #     z = math.floor(
        #         i
        #         / (
        #             constants["SQAURE_GRID_DIMENSION"]
        #             * constants["SQAURE_GRID_DIMENSION"]
        #         ) % n_extracted_datapoints
        #     )
            
        #     pixel_intensities = list(extracted_data[x][y].values())
        #     self.data[i] = torch.cat(
        #         (
        #             pca_data[x][y],
        #             torch.tensor(
        #                 [
        #                     float(fromIndexToLightDir(light_directions_x[z])),
        #                     float(fromIndexToLightDir(light_directions_y[z])),
        #                 ]
        #             ),
        #             torch.tensor([pixel_intensities[z]]),
        #         ),
        #         dim=-1,
        #     )
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx][:-1]
        label = self.data[idx][-1]
        return input, label


class NeuralModel(nn.Module):
    def __init__(self, gaussian_matrix):
        super(NeuralModel, self).__init__()

        self.register_buffer(
            "gaussian_matrix",
            torch.tensor(gaussian_matrix.astype(np.double)),
            persistent=True,
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(constants["PCA_MODEL_INPUT_SIZE"], 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x_light = (6.283185 * (x[:, -2:] @ self.gaussian_matrix)).clone().detach()
        x_light = torch.cat([torch.cos(x_light), torch.sin(x_light)], dim=-1)
        x = torch.cat([x[:, :-2], x_light], dim=-1).float()
        out = self.linear_relu_stack(x)
        return out


def train_pca_model(model_path, extracted_data_file_path, gaussian_matrix, pca_data_file_path):
    print("PCA model: " + model_path)
    print("Training data: " + extracted_data_file_path)

    model = NeuralModel(gaussian_matrix=gaussian_matrix)

    dataset = ExtractedPixelsDataset(extracted_data_file_path, pca_data_file_path)
    dataloader = DataLoader(dataset, batch_size=constants["PCA_BATCH_SIZE"], shuffle=True)

    # Mean Absolute Error
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=constants["PCA_LEARNING_RATE"])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    print("Starting training:")

    for epoch in range(constants["PCA_N_EPOCHS"]):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

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
        loss = running_loss / (constants["SQAURE_GRID_DIMENSION"] * constants["SQAURE_GRID_DIMENSION"])
        print(f"Epoch {epoch + 1}, loss: {loss}, lr: {current_lr}")

    print("Finished Training")

    torch.save(model.state_dict(), model_path)

    print("Saved model to {}".format(model_path))


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
        pca_data_file_path,
        _,
    ) = getChoosenCoinVideosPaths(coin)

    if not os.path.exists(extracted_data_file_path):
        raise (
            Exception(
                "You need to extract the coin data with the analysis before training the model!"
            )
        )

    # gaussian_matrix = np.random.randn(2, 10) * sigma
    if not os.path.exists(constants["GAUSSIAN_MATRIX_FILE_PATH"]):
        gaussian_matrix = generateGaussianMatrix(0, torch.tensor(constants["PCA_SIGMA"]), constants["PCA_H"])
        writeDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"], gaussian_matrix)
    else:
        gaussian_matrix = loadDataFile(constants["GAUSSIAN_MATRIX_FILE_PATH"])


    train_pca_model(model_path, extracted_data_file_path, gaussian_matrix, pca_data_file_path)


if __name__ == "__main__":
    main()
