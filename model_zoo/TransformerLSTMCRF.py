import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .BaseModel import BaseModel

class TransformerLSTMCRF(BaseModel):
    def __init__(self, args):
        super(TransformerLSTMCRF, self).__init__()
        self.transformer = AutoModel.from_pretrained(args.transformer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.transformer_name)
        self.device = torch.device(args.device)

        # layers after transformer
        config = AutoConfig.from_pretrained(args.transformer_name)
        self.dropout = nn.Dropout(args.dropout)
        self.birnn = nn.LSTM(config.hidden_size, args.rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(args.rnn_dim*2, args.num_classes + 2)
        self.crf = CRF(args.num_classes + 2, batch_first=True) # may be updated in the distributed training setting 
        # self.crf = CRF(args.num_classes + 2, use_cuda=True) # may be updated in the distributed training setting 
    
    def forward(self, batch, mode='train', return_predict=False):
        if mode == 'train':
            sents, labels = self.preprocess_input(batch, train=True)
            out = self.transformer(**sents)[0]
            out = self.dropout(out)
            out, _ = self.birnn(out)
            out = self.dropout(out)
            out = self.fc(out)
            loss = self.loss_fn(out, sents['attention_mask'], labels)
            return_item = (out, loss)
            if return_predict:
                return_item += (sents['attention_mask'], labels)
            return return_item
        else:
            sents = self.preprocess_input(batch, train=False)
            out = self.transformer(**sents)[0]
            out = self.dropout(out)
            out, _ = self.birnn(out)
            out = self.dropout(out)
            out = self.fc(out)
            return out, sents['attention_mask']
