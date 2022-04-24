import torch
import numpy as np
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def metric_fn(self, out, mask, label, sent):
        pred = self.predict(out, mask, sent)
        label = label
        acc = []

        # compute accuracy
        for _, (p, l) in enumerate(zip(pred, label)):
            acc.append((torch.tensor(p)==torch.tensor(l)).sum().item()/torch.tensor(l).shape[0])

        # compute the f1
        pred_entities = self.find_entities(data=list(sum(pred, [])))
        truth_entities = self.find_entities(data=list(sum(label, [])))
        true_positive = len(set.intersection(truth_entities, pred_entities))

        if len(truth_entities) != 0:
            precision = float(true_positive) / len(pred_entities) if len(pred_entities) != 0 else 0
            recall = float(true_positive) / len(truth_entities)
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        else:
            precision, recall, f1 = 1, 1, 1

        return np.mean(acc), precision, recall, f1
    
    def find_entities(self, data):
        """ Find all the IOB delimited entities in the data.  Return as a set of (begin, end) tuples. Data is sequence of word, tag pairs. """

        entities = set()

        entityStart = 0
        entityEnd = 0
        
        currentState = "Q0"
        count = 0

        for tag in data:
            count = count + 1
            if currentState == "Q0":
                if tag == 1:
                    currentState = "Q1"
                    entityStart = count
            elif currentState == "Q1":
                if tag == 1:
                    entityEnd = count - 1
                    entities.add((entityStart, entityEnd))
                    entityStart = count
                if tag == 0:
                    entityEnd = count - 1
                    entities.add((entityStart, entityEnd))
                    currentState = "Q0"

        if currentState == "Q1":
            entities.add((entityStart, entityEnd))

        return entities

    def preprocess_input(self, batch, train=False):
        # step 2: change 4 to num + 2
        sents = self.tokenizer(batch['sents'], max_length=512, padding='longest', truncation=True, return_tensors='pt').to(self.device)

        if train:
            # padding -1 to labels
            sent_len = sents['input_ids'].shape[1]
            labels = []
            for i in range(len(batch['labels'])):
                line = self.convert_labels(batch['sents'][i].split(), batch['labels'][i])
                if len(line) >= sent_len:
                    label = line[:sent_len - 1] + [4]
                else:
                    label = line + [4] + [-1] * (sent_len - 1 - len(line))
                labels.append(label)
            labels = torch.LongTensor(labels).to(self.device)

            return sents, labels
        else:
            return sents

    def loss_fn(self, transformer_encode, output_mask, tags):
        loss = -self.crf(emissions=transformer_encode, mask=output_mask.to(torch.uint8), tags=tags, reduction='mean')
        # loss = self.crf.negative_log_loss(transformer_encode, output_mask, tags)
        return loss

    def predict(self, transformer_encode, output_mask, sentences):
        predicts = self.crf.decode(emissions=transformer_encode, mask=output_mask.to(torch.uint8))
        # predicts = self.crf.get_batch_best_path(transformer_encode, output_mask)
        predicts = [self.reconvert_labels(sentences[i].split(), predicts[i]) for i in range(len(predicts))]
        return predicts
    
    def convert_labels(self, sentence, text_labels):
        # step 2: change 4 to num + 1
        labels = []

        for word, label in zip(sentence, text_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return [3] + labels
    
    def reconvert_labels(self, sentence, text_label):
        text_labels = text_label[1:] # remove cls label
        count, spans = 0, []
        for word in sentence:
            tokenized_word = self.tokenizer.tokenize(word)
            spans.append((count, count+len(tokenized_word)-1))
            count = count + len(tokenized_word)
        
        labels = []
        for span in spans:
            if span[0] < len(text_label) - 1:
                labels.append(text_labels[span[0]])
            else:
                labels.append(100)

        return labels

