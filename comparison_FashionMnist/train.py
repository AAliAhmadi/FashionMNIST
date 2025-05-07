import torch
from torch.utils.data import DataLoader

def train_model(model, train_dataset, val_dataset, test_dataset, criterion, optimizer, epochs, batch_size, model_type, device):

    
    print('Processing Neural Network...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    

    for epoch in range(1, epochs+1):
        # ---- Training ----
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            labels = labels.to(device)

            if model_type == 'rnn':
        
                inputs = images.reshape(-1, 28, 28).to(device)
            else:
                inputs = images.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.to(device)

                if model_type == 'rnn':
                    inputs = images.reshape(-1, 28, 28).to(device)
                else:
                    inputs = images.to(device)

                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc*100:.2f}%  "
              f"Val Acc: {val_acc*100:.2f}%")
    
    
    
    
    # ---- Final Test Evaluation ----
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            if model_type == 'rnn':
                inputs = images.reshape(-1, 28, 28).to(device)
            else:
                inputs = images.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print('---------------------------------------')
    return history, test_acc


