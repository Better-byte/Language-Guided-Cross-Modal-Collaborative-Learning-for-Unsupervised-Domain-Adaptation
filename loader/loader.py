import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from loader.jsonloader_no import ImageJSONLoader

def get_transform(is_train):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_clip_transform(is_train):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711))
    ])

def create_data_loader(dataset_name, root_dir, domain, transform, batch_size, cap_src, shuffle=True):
    dataset = ImageJSONLoader(
        root_dir=root_dir,
        json_path=f"metadata/{dataset_name.lower()}.json",
        domain=domain,
        transform=transform,
        return_meta=True,
        _meta_keys=[cap_src]
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

def create_combined_loader(source_loader, pseudo_dataset, batch_size, source_weight=0.7):
    source_size = len(source_loader.dataset)
    pseudo_size = len(pseudo_dataset)
    total_size = source_size + pseudo_size
    
    source_sample_weight = source_weight / source_size
    pseudo_sample_weight = (1 - source_weight) / pseudo_size if pseudo_size > 0 else 0
    
    weights = torch.FloatTensor(
        [source_sample_weight] * source_size + 
        [pseudo_sample_weight] * pseudo_size
    )
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=total_size,
        replacement=True
    )
    
    combined_dataset = ConcatDataset([source_loader.dataset, pseudo_dataset])
    
    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )