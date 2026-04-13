import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertForSequenceClassification

class BertForMultiLabel(nn.Module):
    """BERT model for multilabel text classification."""
    
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        cls_emb = self.dropout(cls_emb)
        logits = self.classifier(cls_emb)
        return logits

class BertForSingleLabel(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    

def load_resnet18_model(num_classes, device):
    """Load ResNet18 with ImageNet pretrained weights."""
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_features, num_classes)
    resnet18.to(device)
    return resnet18


def load_resnet50_model(num_classes, device):
    """Load ResNet50 with ImageNet pretrained weights."""
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_features, num_classes)
    resnet50.to(device)
    return resnet50


def load_densenet121_model(num_classes, device):
    """Load DenseNet121 with ImageNet pretrained weights."""
    densenet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_features = densenet121.classifier.in_features
    densenet121.classifier = nn.Linear(num_features, num_classes)
    densenet121.to(device)
    return densenet121


def load_bert_multilabel_model(num_classes, device, model_name='bert-base-uncased'):
    model = BertForMultiLabel(model_name=model_name, num_labels=num_classes)
    model.to(device)
    return model

def load_bert_singlelabel_model(num_classes, device, model_name='bert-base-uncased'):
    model = BertForSingleLabel(model_name=model_name, num_labels=num_classes)
    model.to(device)
    return model

MODEL_REGISTRY = {
    'resnet18': load_resnet18_model,
    'resnet50': load_resnet50_model,
    'densenet121': load_densenet121_model,
    'bert_multilabel': load_bert_multilabel_model,
    'bert_singlelabel': load_bert_singlelabel_model,
}


def get_model(model_name, num_classes, device):
    """
    Factory function to get a model by name.
    
    Args:
        model_name: Name of the model ('resnet18', 'resnet50', 'densenet121', 'bert_multilabel')
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](num_classes, device)