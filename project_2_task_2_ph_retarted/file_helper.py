import os

import torch


def save_model(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)
    print(f'Model saved to {path}')


def load_model(model, optimizer, path):
    if os.path.isfile(path):
        state = torch.load(path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch']
        print(f'Model loaded from {path}, starting at epoch {start_epoch}')
        return model, optimizer, start_epoch
    else:
        print(f'No model found at {path}, starting from scratch')
        return model, optimizer, 0