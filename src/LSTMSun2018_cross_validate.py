import torch.cuda
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from cross_validator import CrossValidator
from lstm_model_stc import LSTMSun2018
from standard_model_trainer import StandardModelTrainer
from x19_mort_dataset import X19MortalityDataset


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    dataset = X19MortalityDataset()

    model = LSTMSun2018(model_device=cur_device)

    trainer = StandardModelTrainer(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(
            params=model.parameters(), lr=1e-4, betas=(0.5, 0.999)
        ),
    )

    cross_validator = CrossValidator(
        dataset=dataset,
        trainer=trainer,
        num_folds=5,
        batch_size=128,
        epochs_per_fold=25,
        max_global_epochs=5
    )

    cross_validator.run()

    checkpoint_filename = f"{datetime.now()}.tar"
    output_path = (
        Path("/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
             "/training_results")
        / checkpoint_filename
    )

    torch.save(
        obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        },
        f=output_path,
    )


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"Total time = {end - start}")
