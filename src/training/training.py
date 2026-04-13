import torch
from tqdm import tqdm
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def calculate_topk_accuracy(outputs, labels, k_values=[1, 3, 5]):
    """
    Calculate top-k accuracy for given k values.
    
    Args:
        outputs: Model outputs (logits) of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        k_values: List of k values to calculate accuracy for
        
    Returns:
        dict: Dictionary with top-k accuracies for each k value
    """
    _, pred = torch.topk(outputs, max(k_values), dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    
    topk_accuracies = {}
    for k in k_values:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        topk_accuracies[f'top{k}'] = correct_k.item()
    
    return topk_accuracies


def train_vision(model, dataloader, criterion, optimizer, device):
    print(f"Using device: {device}")
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    total_top3_correct = 0
    total_top5_correct = 0

    for batch_data in tqdm(dataloader, desc="Training Batches", leave=False):
        # Handle both 2-tuple and 3-tuple formats (FMoW/GeoYFCCImage return 3-tuple)
        if len(batch_data) == 3:
            inputs, labels, metadata = batch_data
        else:
            inputs, labels = batch_data
            
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate top-k accuracies
        with torch.no_grad():
            topk_acc = calculate_topk_accuracy(outputs, labels, [1, 3, 5])
            total_correct += topk_acc['top1']
            total_top3_correct += topk_acc['top3']
            total_top5_correct += topk_acc['top5']
            total_examples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss, total_examples, total_correct, total_top3_correct, total_top5_correct


def eval_vision(model, dataloader, device):
    model.eval()
    total_examples = 0
    total_correct = 0
    total_top3_correct = 0
    total_top5_correct = 0

    with torch.no_grad():
        for batch_data in dataloader:
            # Handle both 2-tuple and 3-tuple formats (FMoW/GeoYFCCImage return 3-tuple)
            if len(batch_data) == 3:
                inputs, labels, metadata = batch_data
            else:
                inputs, labels = batch_data
                
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # Calculate top-k accuracies
            topk_acc = calculate_topk_accuracy(outputs, labels, [1, 3, 5])
            total_correct += topk_acc['top1']
            total_top3_correct += topk_acc['top3']
            total_top5_correct += topk_acc['top5']
            total_examples += labels.size(0)

    return total_examples, total_correct, total_top3_correct, total_top5_correct


def train_multilabel_text(model, dataloader, criterion, optimizer, device):
    print(f"Using device: {device}")
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    top3_correct = 0
    top5_correct = 0

    for encodings, labels in tqdm(dataloader, desc="Training Batches", leave=False):
        labels = labels.to(device).float()
        
        optimizer.zero_grad()

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
                
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate exact match accuracy (all labels must be correct)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            exact_matches = (preds == labels).all(dim=1).sum().item()
            correct_predictions += exact_matches
            total_predictions += labels.size(0)
            
            # For multilabel, we can still calculate top-k by treating it as single-label
            # using the most confident prediction
            topk_acc = calculate_topk_accuracy(logits, torch.argmax(labels, dim=1), [1, 3, 5])
            top3_correct += topk_acc['top3']
            top5_correct += topk_acc['top5']
    
    avg_loss = total_loss / len(dataloader)
    exact_match_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return total_loss, avg_loss, total_predictions, exact_match_accuracy, top3_correct, top5_correct


def eval_multilabel_text(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    top3_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for encodings, labels in dataloader:
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            labels = labels.to(device).float()

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Calculate exact match accuracy
            exact_matches = (preds == labels).all(dim=1).sum().item()
            correct_predictions += exact_matches
            total_predictions += labels.size(0)
            
            # For multilabel, we can still calculate top-k by treating it as single-label
            # using the most confident prediction
            topk_acc = calculate_topk_accuracy(logits, torch.argmax(labels, dim=1), [1, 3, 5])
            top3_correct += topk_acc['top3']
            top5_correct += topk_acc['top5']

    exact_match_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return total_predictions, exact_match_accuracy, top3_correct, top5_correct

def train_text(model, dataloader, criterion, optimizer, device):
    print(f"Using device: {device}")
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    top3_correct = 0
    top5_correct = 0

    for model_input, labels in tqdm(dataloader, desc="Training Batches", leave=False):
        labels = labels.to(device).long()
        
        optimizer.zero_grad()

        input_ids = model_input["input_ids"].to(device)
        attention_mask = model_input["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate accuracy for single-label classification
        with torch.no_grad():
            # Calculate top-k accuracies
            topk_acc = calculate_topk_accuracy(logits, labels, [1, 3, 5])
            correct_predictions += topk_acc['top1']
            top3_correct += topk_acc['top3']
            top5_correct += topk_acc['top5']
            total_predictions += labels.size(0)

        torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return total_loss, avg_loss, total_predictions, accuracy, top3_correct, top5_correct

def train_text_grad_accumulation(model, dataloader, k_val, criterion, optimizer, device):
    print(f"Using device: {device}")
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    top3_correct = 0
    top5_correct = 0

    #dataloader should have small batch
    optimizer.zero_grad()
    for i, (model_input, labels) in enumerate(tqdm(dataloader, desc="Training Batches", leave=False)):
        labels = labels.to(device).long()

        input_ids = model_input["input_ids"].to(device)
        attention_mask = model_input["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)
        loss.backward()

        if (i+1) % k == 0 or (i+1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        
        # Calculate accuracy for single-label classification
        with torch.no_grad():
            # Calculate top-k accuracies
            topk_acc = calculate_topk_accuracy(logits, labels, [1, 3, 5])
            correct_predictions += topk_acc['top1']
            top3_correct += topk_acc['top3']
            top5_correct += topk_acc['top5']
            total_predictions += labels.size(0)

        torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return total_loss, avg_loss, total_predictions, accuracy, top3_correct, top5_correct


def eval_text(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    top3_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for model_input, labels in dataloader:
            labels = labels.to(device).long()

            input_ids = model_input["input_ids"].to(device)
            attention_mask = model_input["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            
            # Calculate top-k accuracies
            topk_acc = calculate_topk_accuracy(logits, labels, [1, 3, 5])
            correct_predictions += topk_acc['top1']
            top3_correct += topk_acc['top3']
            top5_correct += topk_acc['top5']
            total_predictions += labels.size(0)
            
            torch.cuda.empty_cache()

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return total_predictions, accuracy, top3_correct, top5_correct

def get_train_eval_functions(model_name, grad_accumulation=False):
    if model_name == 'bert_multilabel':
        return train_multilabel_text, eval_multilabel_text
    elif model_name == 'bert_singlelabel':
        if grad_accumulation:
            return train_text_grad_accumulation, eval_text
        return train_text, eval_text
    else:
        return train_vision, eval_vision


def get_criterion(model_name):
    """
    Get appropriate loss function based on model type.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Loss function
    """
    if model_name == 'bert_multilabel':
        return torch.nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss for multilabel (includes sigmoid)
    else:
        return torch.nn.CrossEntropyLoss()  # Original for multiclass