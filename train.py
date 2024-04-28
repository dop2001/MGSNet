import numpy as np
import torch
from utils.loger import Loger, TensorboardWriter
from utils.random_state import RandomState
from utils.dataloader import FeatureDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from models.CNN import CNN, Classifier
from tqdm import tqdm
from utils.metrics import getMetrics
import os


def train(model, train_loader, valid_loader, criterion, optimizer, epoch, device, pth_save_path, loger, summary):
    best_metrics = {'acc': 0}
    for i in range(epoch):
        print(i)
        train_loss_list, train_acc_list = [], []
        train_y_true, train_y_pred = [], []
        model.train()
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            train_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output = output.argmax(axis=1)
            train_y_pred.extend(output.cpu().numpy().tolist())
            train_y_true.extend(label.cpu().numpy().tolist())

        validation_loss_list, validation_acc_list = [], []
        validation_y_true, validation_y_pred = [], []
        model.eval()
        with torch.no_grad():
            for data, label in tqdm(valid_loader):
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)
                validation_loss_list.append(loss.item())
                output = output.argmax(axis=1)
                validation_y_pred.extend(output.cpu().numpy().tolist())
                validation_y_true.extend(label.cpu().numpy().tolist())

            train_loss = sum(train_loss_list) / len(train_loss_list)
            validation_loss = sum(validation_loss_list) / len(validation_loss_list)

            train_metrics = getMetrics(train_y_true, train_y_pred)
            validation_metrics = getMetrics(validation_y_true, validation_y_pred)
            train_metrics['loss'] = train_loss
            validation_metrics['loss'] = validation_loss

            loger.write('epoch:{:02}'.format(i).center(80, '-'))
            loger.write('train | loss={:.3f} | acc={:.3f} | precision={:.3f} | recall={:.3f} | f1={:.3f} | '
                        'specificity={:.3f} | auc={:.3f} | mcc={:.3f}'.format(*train_metrics.values()))
            loger.write('valid | loss={:.3f} | acc={:.3f} | precision={:.3f} | recall={:.3f} | f1={:.3f} | '
                        'specificity={:.3f} | auc={:.3f} | mcc={:.3f}'.format(*validation_metrics.values()))

            if validation_metrics['acc'] > best_metrics['acc']:
                torch.save(model.state_dict(), os.path.join(pth_save_path, "epoch{}_acc{:.6}.pt"
                                                            .format(i, validation_metrics['acc'])))
                best_metrics = validation_metrics

            # write to tensorboard
            record(summary, train_metrics, i + 1)
            record(summary, validation_metrics, i + 1)

    return best_metrics


def record(summary, metrics_dict, count):
    for key, val in metrics_dict.items():
        summary.write(key, {key: val}, count)


if __name__ == '__main__':

    randomState = RandomState(seed=1)
    loger = Loger()
    summary = TensorboardWriter()
    loger.write(message='Training Started'.center(60, '#'))

    configs = {
        'dataset_path': r'./datasets/test1_000.csv',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 8,
        'lr': 1e-4,
        'epoch': 3000,
        'save_path': r'./pths'
    }

    # load dataset
    train_dataset = FeatureDataset(dataset_path=configs['dataset_path'], mode='train')
    valid_dataset = FeatureDataset(dataset_path=configs['dataset_path'], mode='valid')
    loger.write(message='train dataset size is {}'.format(len(train_dataset)))
    loger.write(message='valid dataset size is {}'.format(len(valid_dataset)))

    train_loader = DataLoader(dataset=train_dataset, batch_size=configs['batch_size'], shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=configs['batch_size'], shuffle=False)

    # cnn model
    model = CNN().to(configs['device'])
    # model = Classifier().to(configs['device'])
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # define optimization
    optimizer = torch.optim.Adam(params=model.parameters(), lr=configs["lr"])

    best_metrics = train(model, train_loader, valid_loader, criterion, optimizer, configs['epoch'], configs['device'],
                         configs['save_path'], loger, summary)

    loger.write('best_metrics'.center(80, '-'))
    loger.write(best_metrics)



