import matplotlib.pyplot as plt
import numpy as np

def plot_accuracies(sklearn_results, nn_test_acc):
    import matplotlib.pyplot as plt

    # Combine sklearn and NN results
    results_dict = sklearn_results.copy()
    results_dict['NN'] = nn_test_acc  # Test accuracy

    names = list(results_dict.keys())
    accuracies = [results_dict[name] * 100 for name in names]
    
    

    plt.figure(figsize=(10, 6))
    plt.bar(names, accuracies, color='skyblue')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Classifier Comparison on Test Set')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    plt.show()
    
    
