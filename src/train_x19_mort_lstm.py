from cv_trainer import CrossValidationTrainer
from lstm_model import BinaryBidirectionalLSTM
from x19_mort_dataset import X19MortalityDataset


def main():
    dataset = X19MortalityDataset()
    model = BinaryBidirectionalLSTM(
        input_size=48, lstm_hidden_size=128, fc_hidden_size=32
    )
    cv_trainer = CrossValidationTrainer(
        dataset=dataset,
        model=model,
        num_folds=5,
        batch_size=128,
        epochs_per_fold=2,
        global_epochs=1
    )
    cv_trainer.run()


if __name__ == "__main__":
    main()
