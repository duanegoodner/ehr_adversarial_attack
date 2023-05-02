import torch.cuda
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from lstm_model_stc import LSTMSun2018
from standard_model_trainer import StandardModelTrainer
from weighted_dataloader_builder import WeightedDataLoaderBuilder
from x19_mort_dataset import X19MortalityDataset


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    dataset = X19MortalityDataset()

    model = LSTMSun2018(model_device=cur_device)

    train_dataset_size = int(len(dataset) * 0.8)
    test_dataset_size = len(dataset) - train_dataset_size
    train_dataset, test_dataset = random_split(
        dataset=dataset, lengths=(train_dataset_size, test_dataset_size)
    )
    train_dataloader = WeightedDataLoaderBuilder().build(
        dataset=train_dataset, batch_size=128
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=128, shuffle=True
    )

    trainer = StandardModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(
            params=model.parameters(), lr=1e-4, betas=(0.5, 0.999)
        ),
        save_checkpoints=True,
        checkpoint_dir=Path("/home/duane/dproj/UIUC-DLH/project"
                            "/ehr_adversarial_attack/data"
                            "/train_lstm_sun2018_02"),
        checkpoint_interval=10
    )

    trainer.train_model(num_epochs=3000)


if __name__ == "__main__":
    main()
