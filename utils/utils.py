import os
import torch
import model_zoo

def select_model(args):
    model = model_zoo.__dict__[args.model_name](args)
    return model

def save_checkpoint(model, optimizer, args):
    name = os.path.join(args.save_dir, f'{args.model_name}.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, name)
