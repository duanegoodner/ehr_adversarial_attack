import torch
import torch.nn as nn
from pathlib import Path
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

    data_loader = WeightedDataLoaderBuilder().build(
        dataset=dataset, batch_size=4
    )

    trainer = StandardModelTrainer(
        model=model,
        train_dataloader=data_loader,
        test_dataloader=data_loader,
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

    checkpoint = torch.load(Path("/home/duane/dproj/UIUC-DLH/project"
                                 "/ehr_adversarial_attack/data"
                                 "/training_results/2023-04-30_18:49:09"
                                 ".556432.tar"))
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    metrics = trainer.evaluate_model()
    print(metrics)


if __name__ == "__main__":
    main()
