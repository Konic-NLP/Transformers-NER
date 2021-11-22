from comet_ml import Experiment

import os
import torch
import numpy as np

from tqdm import tqdm
from transformers.optimization import AdamW

from args import args
from datetime import datetime
from utils.logger import init_logger
from utils.dataset import NERDataset, collate_fn
from utils.utils import select_model, save_checkpoint

# global setting
experiment = Experiment(project_name=args.comet_name, api_key=args.comet_key, disabled=(not args.comet_ml))
experiment.log_parameters(vars(args))
running_name = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
logger = init_logger(os.path.join(args.log_dir, running_name+'.log'))

def prepare_data():
    dataset = NERDataset(args)
    train_size = int(args.train_ratio * len(dataset))
    train_data, dev_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    torch.save(dataset, args.dataset)
    torch.save(train_data, args.train_data)
    torch.save(dev_data, args.dev_data)

def train_epoch(model, optimizer, dataloader, epoch):
    model.train()
    pbar = tqdm(total=len(dataloader))

    metric = {'accuracy': [], 'loss': [], 'precision': [], 'recall': [], 'f1': []}

    with experiment.train():
        for step, batch in enumerate(dataloader):
            # compute loss
            out, loss, mask, label = model(batch, return_predict=True)

            # loss backward and backward propgation
            loss.backward()
            optimizer.step()
            # zero the model's grad
            model.zero_grad()

            # compute acc and recall
            acc, precision, recall, f1 = model.metric_fn(out, mask, batch['labels'], batch['sents'])
            metric['accuracy'].append(acc)
            metric['precision'].append(precision)
            metric['recall'].append(recall)
            metric['f1'].append(f1)
            metric['loss'].append(loss.item())

            # update pbar
            pbar.update(1)
            pbar.set_postfix(Train_Epoch=epoch, Loss=loss.item(), Accuracy=acc, Precision=precision, Recall=recall, F1=f1)

            # update experiment
            experiment.log_metrics({'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'loss': loss.item()}, step=step+len(dataloader)*epoch)

    # close pbar
    pbar.close()
    
    # compute the avg in metric
    for item, value in metric.items():
        metric[item] = np.mean(value)
    
    return metric

def test_epoch(model, dataloader, epoch):
    model.eval()
    pbar = tqdm(total=len(dataloader))

    metric = {'accuracy': [], 'loss': [], 'precision': [], 'recall': [], 'f1': []}

    with torch.no_grad():
        for batch in dataloader:
            out, loss, mask, label = model(batch, return_predict=True)

            acc, precision, recall, f1= model.metric_fn(out, mask, batch['labels'], batch['sents'])
            metric['accuracy'].append(acc)
            metric['precision'].append(precision)
            metric['recall'].append(recall)
            metric['f1'].append(f1)
            metric['loss'].append(loss.item())

            # update pbar
            pbar.update(1)
            pbar.set_postfix(Train_Epoch=epoch, Loss=loss.item(), Accuracy=acc, Precision=precision, Recall=recall, F1=f1)

    # close pbar 
    pbar.close()

    # compute the avg in metric
    for item, value in metric.items():
        metric[item] = np.mean(value)
    
    # update experiments
    with experiment.test():
        experiment.log_metrics(metric, step=epoch)
    
    return metric

def train_model():
    # load data
    train_data = torch.load(args.train_data)
    dev_data = torch.load(args.dev_data)
    train_loader, dev_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn), \
        torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # define model, optimizer and loss func
    model = select_model(args).to(args.device)
    # optimizer = AdamW(model.parameters(), lr=args.main_lr)
    transformer_params = [p[-1] for p in list(filter(lambda kv: kv[0].startswith('transformer'), model.named_parameters()))]
    random_init_params = [p[-1] for p in list(filter(lambda kv: not kv[0].startswith('transformer'), model.named_parameters()))]
    optimizer = AdamW([
        {"params": transformer_params, "lr": args.transformer_lr},
        {"params": random_init_params, "lr": args.main_lr},
    ])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.8, last_epoch=-1, verbose=False)

    # training
    best_loss = 1e16
    for epoch in range(args.epochs):
        train_metric = train_epoch(model, optimizer, train_loader, epoch)
        test_metric = test_epoch(model, dev_loader, epoch)
        scheduler.step(test_metric['loss'])

        logger.info('=' * 75)
        logger.info('Epoch: %3d Train Result' % epoch)
        logger.info('Loss: %.2f | Accuracy: %.2f | F1: %.2f | Precision: %.2f | Recall: %.2f' % (
            train_metric['loss'], train_metric['accuracy'], train_metric['f1'], train_metric['precision'], train_metric['recall']
        ))
        logger.info('Epoch: %3d Test Result' % epoch)
        logger.info('Loss: %.2f | Accuracy: %.2f | F1: %.2f | Precision: %.2f | Recall: %.2f' % (
            test_metric['loss'], test_metric['accuracy'], test_metric['f1'], test_metric['precision'], test_metric['recall']
        ))
        logger.info('\n')

        if test_metric['loss'] < best_loss:
            best_loss = test_metric['loss']
            save_checkpoint(model, optimizer, args)

def predict():
    # define id to lable
    label2id = {0: 'O', 1: 'B', 2: 'I', 3: '<START>', 4: '<END>'}

    # define and load model
    model = select_model(args).to(args.device)
    state_dict = torch.load(os.path.join(args.save_dir, f'{args.model_name}.pt'), map_location=torch.device(args.device))
    model.load_state_dict(state_dict['model'])

    dev_data = torch.load(args.dev_data)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, collate_fn=collate_fn)
    predicts = []

    with torch.no_grad():
        for batch in tqdm(dev_loader):
            out, loss, mask, label = model(batch, return_predict=True)
            predict = model.predict(out, mask, batch['sents'])
            predicts += predict

    with open('data/predict_result.txt', 'w') as f:
        for i in range(len(predicts)):
            sent = dev_data[i][0]
            for j, s in enumerate(sent.split()):
                f.write('\t'.join([str(j), s, label2id[predicts[i][j]]]) + '\n')
            f.write('\n')

    with open('data/truth_result.txt', 'w') as f:
        for i in range(len(dev_data)):
            sent = dev_data[i][0]
            for j, s in enumerate(sent.split()):
                f.write('\t'.join([str(j), s, label2id[dev_data[i][-1][j]]]) + '\n')
            f.write('\n')
    
    os.system('python eval.py data/truth_result.txt data/predict_result.txt')

if __name__ == '__main__':
    if args.prepare_data:
        prepare_data()
    
    if args.train:
        train_model()
    
    if args.predict:
        predict()
