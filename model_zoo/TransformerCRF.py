import torch
import torch.nn as nn

from .BaseModel import BaseModel
from torchcrf import CRF

from transformers import AutoTokenizer, AutoModel, AutoConfig

class TransformerCRF(BaseModel):
    def __init__(self, args):
        super(TransformerCRF, self).__init__()
        self.transformer = AutoModel.from_pretrained(args.transformer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.transformer_name)
        self.device = torch.device(args.device)

        # layers after transformer
        config = AutoConfig.from_pretrained(args.transformer_name)
        self.fc = nn.Linear(config.hidden_size, args.num_classes + 2)
        self.dropout = nn.Dropout(args.dropout)
        self.crf = CRF(args.num_classes + 2, batch_first=True) # may be updated in the distributed training setting 
        # self.crf = CRF(args.num_classes + 2, use_cuda=True) # may be updated in the distributed training setting 
    
    def forward(self, batch, mode='train', return_predict=False):
        if mode == 'train':
            sents, labels = self.preprocess_input(batch, train=True)
            out = self.transformer(**sents)[0]
            out = self.dropout(out)
            out = self.fc(out)
            loss = self.loss_fn(out, sents['attention_mask'], labels)
            return_item = (out, loss)
            if return_predict:
                return_item += (sents['attention_mask'], labels)
            return return_item
        else:
            sents, labels = self.preprocess_input(batch, train=False)
            out = self.transformer(**sents)[0]
            out = self.dropout(out)
            out = self.fc(out)
            return out
    
