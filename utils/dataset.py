from torch.utils.data import Dataset

def collate_fn(data):
    batch = {'sents': [], 'labels': []}
    for line in data:
        if type(line) != type(''):
            batch['sents'].append(line[0])
            batch['labels'].append(line[1])
        else:
            batch['sents'].append(line)
    return batch


class NERDataset(Dataset):
    def __init__(self, args, mode='train'):
        # self.label2id = {'O': 1, 'B': 2, 'I': 3} # think more, what is padding idx
        # step 1: change here
        self.label2id = {'O': 0, 'B': 1, 'I': 2} # think more, what is padding idx
        raw_file = args.raw_file if mode == 'train' else args.predict_file
        self.mode = mode
        self.texts, self.labels = self.process_raw_data(raw_file, mode=mode)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.texts[index], self.labels[index]
        else:
            return self.texts[index]

    def process_raw_data(self, raw_file, mode):
        # extracting data from raw file
        with open(raw_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        texts, labels = [], []
        cache = []
        for line in lines:
            line = line.strip()
            if len(line) != 0:
                cache.append(line.split('\t'))
            else:
                texts.append(' '.join([c[0] for c in cache]))
                if mode == 'train':
                    labels.append([self.label2id[c[1]] for c in cache])
                cache = []
        
        if mode == 'train':
            assert len(texts) == len(labels)
        
        return texts, labels
