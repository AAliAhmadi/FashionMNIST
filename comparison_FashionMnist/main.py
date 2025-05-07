import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from model import FashionRNN, SimpleCNN
from train import train_model
from sklearn_baselines import sklearn_comparison
from utils import plot_accuracies

def main(args):
    # Data loading
    transform = transforms.ToTensor()
    full_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)

    # Train/validation split
    val_size = int(len(full_train) * args.val_split)
    train_size = len(full_train) - val_size
    train_data, val_data = random_split(full_train, [train_size, val_size])

    # Model selection
    if args.model == 'rnn':
        model = FashionRNN(
            input_size=28, hidden_size=args.hidden_size,
            num_layers=args.num_layers, num_classes=10
        ).to(args.device)
    else:
        model = SimpleCNN(num_classes=10).to(args.device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    history, nn_test_acc = train_model(
        model, train_data, val_data, test_data,
        criterion, optimizer,
        args.epochs, args.batch_size,
        args.model, args.device
    )


    # Sklearn baselines
    sklearn_results = sklearn_comparison(full_train, test_data, use_svm=args.use_svm)



    # Plot
 
    sklearn_results['NeuralNet'] = nn_test_acc

    plot_accuracies(sklearn_results, nn_test_acc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rnn','cnn'], default='rnn',
                        help="Model to train: 'rnn' or 'cnn'")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--device', type=str,
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
                        
    parser.add_argument('--use_svm', action='store_true',
                    help="Include SVM in sklearn baselines (slow)")

    args = parser.parse_args()
    main(args)
