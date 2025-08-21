import math
import torch
import torch.nn.functional as F
from .entropy_loss import EntropyLoss
from .consistency_loss import ConsistencyLoss
from .feature_alignment_loss import FeatureAlignmentLoss
from .cosim_alignment_loss import CosimAlignmentLoss
from .contrastive_loss import ContrastiveLoss
from .triplet_loss import TripletLoss

class MultiTaskLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, lambda_cons=1.0, lambda_ent=0.1,
                 lambda_align=0.5, lambda_contra=0.3, use_cosim=True,
                 use_triplet=True, total_steps=10000):
        super(MultiTaskLoss, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones([]) * temperature)
        self.lambda_cons = lambda_cons
        self.lambda_ent = lambda_ent
        self.lambda_align = lambda_align
        self.lambda_contra = lambda_contra
        self.use_cosim = use_cosim
        self.use_triplet = use_triplet
        self.current_step = 0
        self.total_steps = total_steps
        
        # 初始化子损失模块
        self.entropy_loss = EntropyLoss()
        self.consistency_loss = ConsistencyLoss()
        self.feature_alignment_loss = FeatureAlignmentLoss()
        self.cosim_alignment_loss = CosimAlignmentLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.triplet_loss = TripletLoss()

    def forward(self, outputs, labels=None, is_source=True):
        self.current_step += 1
        progress = self.current_step / self.total_steps

        # 余弦退火动态权重调整
        lambda_align = self.lambda_align * (0.5 * (1 + math.cos(math.pi * progress)))
        lambda_contra = self.lambda_contra * (0.5 * (1 + math.cos(math.pi * progress)))

        if is_source:
            labels = labels.long().squeeze()

            # 分类损失
            text_cls_loss = F.cross_entropy(outputs['text_logits'], labels)
            vision_cls_loss = F.cross_entropy(outputs['vision_logits'], labels)
            final_cls_loss = F.cross_entropy(outputs['final_logits'], labels) * 1.2

            # 一致性损失
            cons_loss = self.consistency_loss(
                outputs['text_logits'],
                outputs['vision_logits'],
                outputs['final_logits']
            )

            # 特征对齐损失
            align_loss = self.feature_alignment_loss(
                outputs['text_attn'],
                outputs['vision_attn']
            )

            # 余弦相似度对齐损失（可选）
            if self.use_cosim and 'text_seq' in outputs and 'vision_seq' in outputs:
                align_loss += self.cosim_alignment_loss(outputs['text_seq'], outputs['vision_seq'])

            # 对比损失
            contra_loss = self.contrastive_loss(
                outputs['text_features'],
                outputs['vision_features'],
                labels
            )

            # 总损失计算
            total_loss = text_cls_loss + vision_cls_loss + final_cls_loss + \
                         self.lambda_cons * cons_loss + \
                         lambda_align * align_loss + \
                         lambda_contra * contra_loss

            # 三元组损失（可选）
            if self.use_triplet:
                triplet = self.triplet_loss(outputs['text_features'], outputs['vision_features'])
                total_loss += 0.3 * triplet

        else:
            # 无监督域适应损失
            cons_loss = self.consistency_loss(
                outputs['text_logits'],
                outputs['vision_logits'],
                outputs['final_logits']
            )

            # 熵最小化损失
            ent_loss = (self.entropy_loss(outputs['text_logits']) +
                        self.entropy_loss(outputs['vision_logits']) +
                        2 * self.entropy_loss(outputs['final_logits'])) / 4.0

            # 特征对齐损失
            align_loss = self.feature_alignment_loss(
                outputs['text_attn'],
                outputs['vision_attn']
            )

            # 余弦相似度对齐损失（可选）
            if self.use_cosim and 'text_seq' in outputs and 'vision_seq' in outputs:
                align_loss += self.cosim_alignment_loss(outputs['text_seq'], outputs['vision_seq'])

            # 对比损失
            contra_loss = self.contrastive_loss(
                outputs['text_features'],
                outputs['vision_features']
            )

            # 总损失计算
            total_loss = self.lambda_cons * cons_loss + \
                         self.lambda_ent * ent_loss + \
                         lambda_align * align_loss + \
                         lambda_contra * contra_loss

            # 三元组损失（可选）
            if self.use_triplet:
                triplet = self.triplet_loss(outputs['text_features'], outputs['vision_features'])
                total_loss += 0.3 * triplet

        return total_loss