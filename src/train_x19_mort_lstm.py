import torch.cuda
from datetime import datetime
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader

from cv_trainer import CrossValidationTrainer
from lstm_model import BinaryBidirectionalLSTM
from lstm_model_from_sc import LSTMSun2018
from x19_mort_dataset import X19MortalityDataset


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    dataset = X19MortalityDataset()
    # model = BinaryBidirectionalLSTM(
    #     device=device, input_size=48, lstm_hidden_size=128, fc_hidden_size=32
    # )
    model = LSTMSun2018(device=device)
    model.set_loss_fn(loss=nn.CrossEntropyLoss()).set_optimizer(
        torch.optim.Adam(params=model.parameters())
    )

    cv_trainer = CrossValidationTrainer(
        device=device,
        dataset=dataset,
        model=model,
        num_folds=5,
        batch_size=128,
        epochs_per_fold=1,
        global_epochs=5,
    )
    cv_trainer.run()

    final_eval_loader = DataLoader(
        dataset=dataset, batch_size=128, shuffle=True
    )

    model.evaluate_model(test_loader=final_eval_loader)

    checkpoint_filename = f"{datetime.now()}.tar"
    output_path = (
        Path("/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data")
        / checkpoint_filename
    )

    # torch.save(
    #     obj={
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": model.optimizer.state_dict(),
    #     },
    #     f=output_path,
    # )


if __name__ == "__main__":
    main()
