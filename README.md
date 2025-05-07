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
git clone https://github.com/yourusername/fashion-mnist-comparison.git
cd fashion-mnist-comparison
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Run the training script

```bash
python main.py --model cnn --epochs 10
```

Add other optional arguments:

```bash
--model {cnn,rnn} --batch_size 64 --hidden_size 128 --num_layers 2 --lr 0.001 --device cpu
```

### 4. Evaluate sklearn classifiers

This can be done in the script using:

```python
from sklearn_baselines import sklearn_comparison
results = sklearn_comparison(train_dataset, test_dataset)
```

### 5. Plotting results

You can visualize training progress and final classifier comparisons using:

```python
from utils import plot_accuracies
plot_accuracies(history, results)
```

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
