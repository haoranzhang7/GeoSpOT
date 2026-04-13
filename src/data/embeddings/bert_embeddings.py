import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from typing import List

from geoyfcc import GeoYFCCText 


def collate_fn(batch, tokenizer):
    text = [item["combined_text"] for item in batch]
    encodings = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # Store labels as a list of lists (multi-label)
    return encodings


def generate_bert_embeddings(
    dataset: GeoYFCCText,
    batch_size: int = 32,
    device: str = None
) -> torch.Tensor:
    """
    Generate BERT embeddings for the 'combined_text' field of GeoYFCCText dataset.
    
    Args:
        dataset: GeoYFCCText dataset object
        batch_size: batch size for DataLoader
        device: 'cuda' or 'cpu'; auto-detect if None
    Returns:
        embeddings: torch.Tensor of shape (num_samples, hidden_size)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # DataLoader for batching with custom collate function
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    all_embeddings = []
    with torch.no_grad():
        for encodings in tqdm(loader, desc="Generating BERT embeddings"):
            encodings = {k: v.to(device) for k, v in encodings.items()}
            outputs = model(**encodings)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())


    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"[INFO] Generated embeddings shape: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    dataset = GeoYFCCText(root="../../../data/geoyfcc")
    print(f"[INFO] Dataset loaded with {len(dataset)} samples")

    embeddings = generate_bert_embeddings(dataset, batch_size=64)
    # Save embeddings to disk
    torch.save(embeddings, "../../../data/geoyfcc/geoyfcc_bert_embeddings.pt")
    print("[INFO] Embeddings saved to geoyfcc_bert_embeddings.pt")
