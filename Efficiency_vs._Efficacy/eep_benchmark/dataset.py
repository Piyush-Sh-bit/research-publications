import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class DocumentDataset(Dataset):
    """
    A PyTorch Dataset for loading Document Intelligence images and annotations.
    Expects a JSON Lines (JSONL) file or JSON with:
    [
        {"image_path": "images/doc1.png", "question": "What is the total?", "answer": "$500", "task": "Key-Value", "script": "Latin"},
        ...
    ]
    """
    def __init__(self, annotation_path, base_image_dir, processor=None, transform=None):
        self.base_image_dir = base_image_dir
        self.processor = processor
        self.transform = transform
        
        self.samples = []
        with open(annotation_path, "r", encoding="utf-8") as f:
            if annotation_path.endswith('.jsonl'):
                for line in f:
                    self.samples.append(json.loads(line))
            else:
                self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.base_image_dir, sample["image_path"])
        
        # Load image (convert to RGB to unify formats)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "id": sample.get("id", str(idx)),
            "image": image,
            "question": sample["question"],
            "ground_truth": str(sample["answer"]),
            "task": sample.get("task", "Unknown"),
            "script": sample.get("script", "Latin")
        }

def get_dataloader(annotation_path, base_image_dir, batch_size=8, processor=None, num_workers=4):
    """Returns a DataLoader for batched inference."""
    dataset = DocumentDataset(annotation_path, base_image_dir, processor)
    
    def collate_fn(batch):
        # Custom collate depending on whether processor is provided.
        # For MLLMs, it's often easier to yield lists of PIL images and let the inference loop process them.
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        gts = [item["ground_truth"] for item in batch]
        tasks = [item["task"] for item in batch]
        scripts = [item["script"] for item in batch]
        return {"images": images, "questions": questions, "gts": gts, "tasks": tasks, "scripts": scripts}

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
