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



## 📊 Example Output

Include example plots or screenshots here.

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
