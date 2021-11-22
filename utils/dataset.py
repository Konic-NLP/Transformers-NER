from torch.utils.data import Dataset

def collate_fn(data):
    batch = {'sents': [], 'labels': []}
    for line in data:
        batch['sents'].append(line[0])
        batch['labels'].append(line[1])
    return batch

class NERDataset(Dataset):
    def __init__(self, args, mode='train'):
        # self.label2id = {'O': 1, 'B': 2, 'I': 3} # think more, what is padding idx
        self.label2id = {'O': 0, 'B': 1, 'I': 2} # think more, what is padding idx
        self.texts, self.labels = self.process_raw_data(args.raw_file)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def process_raw_data(self, raw_file):
        # extracting data from raw file
        with open(raw_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        texts, labels = [], []
        cache = []
        for line in lines:
            line = line.strip()
            if len(line) != 0:
                cache.append(line.split('\t')[1:])
            else:
                texts.append(' '.join([c[0] for c in cache]))
                labels.append([self.label2id[c[1]] for c in cache])
                cache = []
        
        assert len(texts) == len(labels)
        
        return texts, labels
