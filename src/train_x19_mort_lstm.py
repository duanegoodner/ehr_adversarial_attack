import torch.cuda

from cv_trainer import CrossValidationTrainer
from lstm_model import BinaryBidirectionalLSTM
from x19_mort_dataset import X19MortalityDataset


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    dataset = X19MortalityDataset()
    model = BinaryBidirectionalLSTM(
        device=device, input_size=48, lstm_hidden_size=128, fc_hidden_size=32
    )
    cv_trainer = CrossValidationTrainer(
        device=device,
        dataset=dataset,
        model=model,
        num_folds=5,
        batch_size=128,
        epochs_per_fold=5,
        global_epochs=3
    )
    cv_trainer.run()

    


if __name__ == "__main__":
    main()
