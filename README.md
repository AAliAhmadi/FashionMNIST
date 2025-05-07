# Fashion-MNIST Classifier Comparison

This project compares the performance of deep learning models (CNN/RNN) with classic machine learning classifiers (SVM, Decision Tree, Random Forest, etc.) on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

## 📦 Features

* Train deep learning models using PyTorch (CNN or RNN)
* Evaluate model accuracy on validation and test datasets
* Compare against classic scikit-learn classifiers
* Visualize training progress and accuracy comparison

## 🧠 Models

### Neural Networks

* CNN (Convolutional Neural Network)
* RNN (Recurrent Neural Network)

### Classical Classifiers

* SVM
* Decision Tree
* Random Forest
* Naive Bayes
* Linear Discriminant Analysis (LDA)
* AdaBoost

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AAliAhmadi/fashion-mnist-comparison.git
cd fashion-mnist-comparison
```

### 2. 📦 Requirements
- Python 3.7+

- PyTorch

- torchvision

- scikit-learn

- matplotlib

Install dependencies with:

```bash
pip install torch torchvision scikit-learn matplotlib
```

### 3. Run the training script

#### 3.1 🧪 SVM Usage
By default, SVM is disabled due to its high runtime. To include it in the comparison:
```bash
python main.py --use_svm
```

⚠️ Warning: Training SVM on the full dataset can be slow, especially on CPU.


**Add other optional arguments:**

```bash
python main.py --model cnn --epochs 20 --batch_size 128 

```

**All available options**
| Argument        | Description                               | Default                   |
| --------------- | ----------------------------------------- | ------------------------- |
| `--model`       | Choose between `'cnn'` or `'rnn'`         | `'rnn'`                   |
| `--epochs`      | Number of training epochs                 | `10`                      |
| `--batch_size`  | Training batch size                       | `64`                      |
| `--hidden_size` | Hidden size for RNN                       | `128`                     |
| `--num_layers`  | Number of RNN layers                      | `2`                       |
| `--lr`          | Learning rate                             | `1e-3`                    |
| `--val_split`   | Fraction of training data used as dev set | `0.1`                     |
| `--device`      | `'cuda'` or `'cpu'`                       | auto-detects              |
| `--use_svm`     | Include SVM in sklearn comparison         | optional, default `False` |

## ✅ Alternative Running option: 
Just run ```run_main.ipynb``` file in your notebook!

## 📊 Example Output

```
Processing Neural Network...
Epoch 1/5  Train Loss: 0.5768  Train Acc: 79.27%  Val Acc: 86.68%
Epoch 2/5  Train Loss: 0.3765  Train Acc: 86.54%  Val Acc: 89.38%
Epoch 3/5  Train Loss: 0.3214  Train Acc: 88.47%  Val Acc: 90.23%
Epoch 4/5  Train Loss: 0.2903  Train Acc: 89.65%  Val Acc: 91.05%
Epoch 5/5  Train Loss: 0.2663  Train Acc: 90.36%  Val Acc: 91.23%
Test Accuracy: 90.49%
--------------------------------------
Processing DecisionTree...
DecisionTree: 78.88%
---------------------------------------
Processing NaiveBayes...
NaiveBayes: 58.56%
---------------------------------------
Processing LDA...
LDA: 81.51%
---------------------------------------
Processing RandomForest...
RandomForest: 87.34%
---------------------------------------
Processing AdaBoost...
AdaBoost: 54.25%
---------------------------------------
```

![download (1)](https://github.com/user-attachments/assets/ab710fa3-078c-41d6-b874-fa7e5ef48b54)


## 📝 Notes

* The training set is split into training and validation (e.g., 90/10).
* All classifiers are trained on the same training subset for fair comparison.
* Sklearn models require flattened inputs; ensure preprocessing matches across models.

## 📁 Project Structure

```
.
├── main.py                # Entry point to train NN models
├── sklearn_baselines.py   # Sklearn classifier training and evaluation
├── utils.py               # Visualization functions
├── model.py               # CNN/RNN model definitions
├── train.py               # NN training logic
├── data/                  # Downloaded Fashion-MNIST data
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📜 License

This project is licensed under the MIT License.
