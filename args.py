from argparse import ArgumentParser

parser = ArgumentParser()

# running mode
parser.add_argument('--prepare_data', action='store_true', help='extracting infomation from raw data file')
parser.add_argument('--train', action='store_true', help='training progress')
parser.add_argument('--predict', action='store_true', help='predicting progress')

# parameters in comet ml (a toolkit to visualize the process of training)
parser.add_argument('--comet_ml', action='store_true', help='indicate if use comet ml')
parser.add_argument('--comet_key', type=str, default='JXqjTyjl3HyEnlLnosq7FUgH1', help='API key for Sen\'s comet account')
parser.add_argument('--comet_name', type=str, default='BoulderNLPCourse', help='repo name in comet ml')
parser.add_argument('--ex_name', type=str, default='', help='experiment name in comet ml')

# parameters about data
parser.add_argument('--raw_file', type=str, default='data/S21-gene-train.txt', help='path to the raw data file')
parser.add_argument('--dataset', type=str, default='data/dataset.pt', help='path to the processed dataset')
parser.add_argument('--train_data', type=str, default='data/train_data.pt', help='path to the processed train dataset')
parser.add_argument('--dev_data', type=str, default='data/dev_data.pt', help='path to the processed dev dataset')
parser.add_argument('--train_ratio', type=float, default=0.8, help='num ratio of the training data')

# parameters about model
parser.add_argument('--model_name', type=str, default='TransformerCRF', choices=['TransformerCRF', 'TransformerLSTMCRF'], help='name of the used model for NER')
parser.add_argument('--transformer_name', type=str, default='bert-base-uncased', help='name of the pretrained transformer model')
parser.add_argument('--num_classes', type=int, default=3, help='numbers of the predicted classes')
parser.add_argument('--rnn_dim', type=int, default=128, help='rnn hidden dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate of model')

# parameters about training
parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
parser.add_argument('--epochs', type=int, default=10, help='epochs to train')
parser.add_argument('--transformer_lr', type=float, default=1e-5, help='learning rate of optimizer')
parser.add_argument('--main_lr', type=float, default=1e-3, help='learning rate of optimizer')
parser.add_argument('--scheduler_step', type=float, default=600, help='learning rate of optimizer')
parser.add_argument('--device', type=str, default='cuda', help='device to run')

# parameters about logging and saving
parser.add_argument('--log_dir', type=str, default='log', help='directory of the logger file')
parser.add_argument('--save_dir', type=str, default='save_temp', help='directory of the save checkpoints')

args = parser.parse_args()
