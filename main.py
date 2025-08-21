import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import logging
import torch
import clip
import json
from transformers import BertForSequenceClassification, AutoTokenizer

from config.args import parse_args, DATASET_CLASSES
from config.paths import setup_paths
from utils.logging import setup_logging
from utils.seed import set_seed
from data.loader import get_transform, create_data_loader, create_combined_loader
from models.fusion import DualBranchNetwork
from models.clip_encoder import CLIPImageEncoder
from utils.pseudo_labels import generate_pseudo_labels
from train.trainer import train
from train.evaluate import evaluate_target_accuracy

def main():
    args = parse_args()
    paths = setup_paths(args)
    
    logger = setup_logging(paths['log_file'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(args.seed)
    
    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Device: {device}")
    
    try:
        transform = get_transform(is_train=True)
        source_loader = create_data_loader(args.dataset, args.root_dir, args.source, 
                                          transform, args.batch_size, args.cap_src)
        target_loader = create_data_loader(args.dataset, args.root_dir, args.target, 
                                          transform, args.batch_size, args.cap_src)
        

        num_classes = DATASET_CLASSES[args.dataset]
        

        clip_model, _ = clip.load("ViT-B/16", device=device)
        clip_encoder = CLIPImageEncoder(clip_model).to(device)
        for param in clip_encoder.parameters():
            param.requires_grad = False
            

        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        text_model = BertForSequenceClassification.from_pretrained(
            args.backbone,
            num_labels=num_classes,
            output_hidden_states=True
        ).to(device)
        
  
        checkpoint_dir = os.path.join("checkpointsbase", f"{args.dataset}_{args.source}2{args.target}")
        text_model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "best_text.pth")),
            strict=False
        )
        
    
        freeze_layers = 8
        for layer_idx in range(freeze_layers):
            for param in text_model.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

        fusion_model = DualBranchNetwork(
            text_dim=768,
            vision_dim=512,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout
        ).to(device)
        
  
        train(args, device, text_model, clip_encoder, fusion_model, 
             tokenizer, source_loader, target_loader, paths)
        
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
