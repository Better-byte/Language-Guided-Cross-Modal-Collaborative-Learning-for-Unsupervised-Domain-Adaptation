# Language-Guided-Cross-Modal-Collaborative-Learning-for-Unsupervised-Domain-Adaptation
## Overview​​
This framework proposes an innovative Unsupervised Domain Adaptation (UDA) approach that significantly enhances cross-domain generalization capabilities through language-guided cross-modal collaborative learning. The core concept leverages the inherent property of textual descriptions exhibiting reduced domain gaps to facilitate knowledge transfer. By integrating a bidirectional Cross-modal Gating Learning Module (CGLM) and a Cross-attention Multimodal Fusion Module (CMFM), the framework achieves domain-invariant feature learning, effectively decoupling domain-specific artifacts from semantic representations.<img width="2004" height="1080" alt="framework" src="https://github.com/user-attachments/assets/e585ae56-aac3-47ef-a4ca-d57b0140de24" />
## Prepare

### Dataset
For non-annotated datasets use ​​BLIP-2​​ to generate image captions
```bash
from transformers import Blip2Processor, Blip2ForConditionalGeneration  
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")  
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b") 
```
## Acknowledgments

This project builds upon the invaluable contributions of following open-source projects:

1.DAMP (https://github.com/TL-UESTC/DAMP)

2.LaGTran (https://tarun005.github.io/lagtran/)
