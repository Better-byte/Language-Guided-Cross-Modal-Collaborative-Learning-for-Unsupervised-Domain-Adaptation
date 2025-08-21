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
    # 解析参数并设置路径
    args = parse_args()
    paths = setup_paths(args)
    
    # 设置日志
    logger = setup_logging(paths['log_file'])
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 记录参数
    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Device: {device}")
    
    try:
        # 加载数据
        transform = get_transform(is_train=True)
        source_loader = create_data_loader(args.dataset, args.root_dir, args.source, 
                                          transform, args.batch_size, args.cap_src)
        target_loader = create_data_loader(args.dataset, args.root_dir, args.target, 
                                          transform, args.batch_size, args.cap_src)
        
        # 初始化模型
        num_classes = DATASET_CLASSES[args.dataset]
        
        # CLIP模型
        clip_model, _ = clip.load("ViT-B/16", device=device)
        clip_encoder = CLIPImageEncoder(clip_model).to(device)
        for param in clip_encoder.parameters():
            param.requires_grad = False
            
        # 文本模型
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        text_model = BertForSequenceClassification.from_pretrained(
            args.backbone,
            num_labels=num_classes,
            output_hidden_states=True
        ).to(device)
        
        # 加载预训练权重
        checkpoint_dir = os.path.join("checkpointsbase", f"{args.dataset}_{args.source}2{args.target}")
        text_model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "best_text.pth")),
            strict=False
        )
        
        # 冻结部分层
        freeze_layers = 8
        for layer_idx in range(freeze_layers):
            for param in text_model.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
        
        # 跨模态融合模型
        fusion_model = DualBranchNetwork(
            text_dim=768,
            vision_dim=512,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout
        ).to(device)
        
        # 训练
        train(args, device, text_model, clip_encoder, fusion_model, 
             tokenizer, source_loader, target_loader, paths)
        
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()