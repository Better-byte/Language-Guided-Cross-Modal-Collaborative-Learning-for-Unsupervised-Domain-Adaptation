import os
import json
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
from loader.jsonloader_no import ImageJSONLoader
import logging

def generate_pseudo_labels(text_model, clip_model, fusion_model, target_loader, 
                         tokenizer, threshold, device, args, paths, stage='early'):
    text_model.eval()
    if fusion_model is not None:
        fusion_model.eval()
    
    original_json_path = f"metadata/{args.dataset.lower()}.json"
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    
    pseudo_data = {"categories": original_data["categories"]}
    for domain in original_data.keys():
        if domain != "categories":
            pseudo_data[domain] = {"images": [], "annotations": [], "metadata": []}

    with torch.no_grad():
        for batch in tqdm(target_loader, desc=f"Generating {stage} pseudo-labels"):
            image_ids, images, _, metadata = batch
            captions = metadata[args.cap_src]
            images = images.to(device)
        
            with torch.cuda.amp.autocast():
                clip_features = clip_model(images)
                tokens = tokenizer(captions, padding="longest", truncation=True, return_tensors="pt")
                tokens = {k: v.to(device) for k, v in tokens.items()}
                
                if stage == 'early':
                    text_outputs = text_model(**tokens)
                    text_probs = F.softmax(text_outputs.logits, dim=-1)
                    max_probs, preds = torch.max(text_probs, dim=1)
                else:
                    text_outputs = text_model(**tokens)
                    text_features = text_outputs.hidden_states[-1][:, 0, :]
                    outputs = fusion_model(text_features, clip_features)
                    final_probs = F.softmax(outputs['final_logits'], dim=-1)
                    max_probs, preds = torch.max(final_probs, dim=1)
            
            mask = max_probs >= threshold
            
            if mask.sum() > 0:
                valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
                for idx in valid_indices:
                    image_id = image_ids[idx.item()]
                    caption = captions[idx.item()]
                    pred = preds[idx].item()
                    prob = max_probs[idx].item()
                    
                    original_image_info = next(
                        (img for img in original_data[args.target]["images"] if img["id"] == image_id), 
                        None
                    )
                    original_metadata_info = next(
                        (meta for meta in original_data[args.target]["metadata"] if meta["image_id"] == image_id),
                        None
                    )
                    
                    if original_image_info and original_metadata_info:
                        category_name = next(
                            (cat["category_name"] for cat in original_data["categories"] 
                             if cat["category_id"] == pred),
                            f"class_{pred}"
                        )
                        pseudo_data[args.target]["images"].append(original_image_info)
                        pseudo_data[args.target]["annotations"].append({
                            "image_id": image_id,
                            "category": pred,
                            "class_name": category_name
                        })
                        new_metadata = original_metadata_info.copy()
                        new_metadata["file_tag"] = f"{args.target}_{pred}_{image_id}"
                        pseudo_data[args.target]["metadata"].append(new_metadata)
    
    if len(pseudo_data[args.target]["images"]) == 0:
        logging.warning("No pseudo-labels generated in this iteration")
        return ImageJSONLoader(empty=True), 0.0, None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(paths['pseudo_labels'], f"pseudo_{stage}_{timestamp}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pseudo_data, f, indent=4)
    
    coverage = len(pseudo_data[args.target]["images"]) / len(target_loader.dataset)
    logging.info(f"Generated {len(pseudo_data[args.target]['images'])} pseudo-labels ({coverage:.2f} coverage)")
    
    pseudo_dataset = ImageJSONLoader(
        root_dir=args.root_dir,
        json_path=output_path,
        domain=args.target,
        transform=target_loader.dataset.transform,
        return_meta=True,
        _meta_keys=[args.cap_src]
    )
    
    return pseudo_dataset, coverage, output_path