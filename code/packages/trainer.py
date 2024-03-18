import torch
import os
from packages.utils import *
from packages import Models
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import logging
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(Models, hyp_params.model)(hyp_params)

    if hyp_params.use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    optimizer = getattr(optim, hyp_params.optim)(
        model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.weight_decay)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader, device)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader, device):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']
    os.makedirs('./log/', exist_ok=True)
    log_dir = './log/'
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=f'./log/train_{hyp_params.fold_num}.log', encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)
    writer = SummaryWriter(log_dir)

    def train(model, optimizer, criterion, device):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train//hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        i_batch = 0
        for labels, strs, toks in train_loader:
            optimizer.zero_grad()
            i_batch += 1
            toks.to(device)
            y = torch.tensor(labels, device=device).float().unsqueeze(-1)

            batch_size = hyp_params.batch_size
            logits = model(strs, toks)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()

        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, device, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        model.to(device)
        results = []
        truths = []
        with torch.no_grad():
            for labels, strs, toks in loader:
                toks.to(device)
                y = torch.tensor(labels, device=device).float().unsqueeze(-1)

                batch_size = hyp_params.batch_size

                logits = model(strs, toks)
                total_loss += criterion(logits, y).item() * batch_size

                results.append(torch.sigmoid(logits))
                truths.append(y)
        avg_loss = total_loss / \
            (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_f1 = 0
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train(model, optimizer, criterion, device)
        val_loss, results, truths = evaluate(model, criterion, device, test=False)

        val_metrics = metrics(results, truths)
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        val_acc = val_metrics['Accuracy']
        val_f1 = val_metrics['F1-score']

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Valid Acc {:.2f} | Valid F1 {:.3f}'.format(
            epoch, duration, val_loss, val_acc*100, val_f1))
        print("-"*50)

        writer.add_scalar('Valid/Loss', val_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar('Valid/' + key, value, epoch)

        if val_f1 > best_f1:
            print(f"Saved model at {hyp_params.name}_{hyp_params.model}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_f1 = val_f1

    writer.close()
    model = load_model(hyp_params, name=hyp_params.name)
    if hyp_params.kfold != 0:
        _, results, truths = evaluate(model, criterion, device, test=False)
    else:
        _, results, truths = evaluate(model, criterion, device, test=True)

    metrics_dict = metrics(results, truths)
    print(metrics_dict)
    print("-"*50)

    return best_f1

def final_train(hyp_params, train_loader, test_loader):
    model = getattr(Models, hyp_params.model)(hyp_params)

    if hyp_params.use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    optimizer = getattr(optim, hyp_params.optim)(
        model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.weight_decay)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return final_train_model(settings, hyp_params, train_loader, test_loader, device)

def final_train_model(settings, hyp_params, train_loader, test_loader, device):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']
    os.makedirs('./log/', exist_ok=True)
    log_dir = './log/'
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=f'./log/train.log', encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)
    writer = SummaryWriter(log_dir)

    def train(model, optimizer, criterion, device):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train//hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        i_batch = 0
        for labels, strs, toks in train_loader:
            optimizer.zero_grad()
            i_batch += 1
            toks.to(device)
            y = torch.tensor(labels, device=device).float().unsqueeze(-1)

            batch_size = hyp_params.batch_size
            logits = model(strs, toks)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()

        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, device):
        model.eval()
        loader = test_loader
        total_loss = 0.0

        model.to(device)
        results = []
        truths = []
        with torch.no_grad():
            for labels, strs, toks in loader:
                toks.to(device)
                y = torch.tensor(labels, device=device).float().unsqueeze(-1)

                batch_size = hyp_params.batch_size

                logits = model(strs, toks)
                total_loss += criterion(logits, y).item() * batch_size

                results.append(torch.sigmoid(logits))
                truths.append(y)
        avg_loss = total_loss / \
            hyp_params.n_test

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths
    
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train(model, optimizer, criterion, device)
        val_loss, results, truths = evaluate(model, criterion, device)

        val_metrics = metrics(results, truths)
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)

        val_acc = val_metrics['Accuracy']
        val_f1 = val_metrics['F1-score']

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Test Loss {:5.4f} | Test Acc {:.2f} | Test F1 {:.3f}'.format(
            epoch, duration, val_loss, val_acc*100, val_f1))
        print("-"*50)

        writer.add_scalar('Valid/Loss', val_loss, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar('Valid/' + key, value, epoch)

        print(f"Saved model at {hyp_params.name}_{hyp_params.model}.pt!")
        save_model(hyp_params, model, name=hyp_params.name)

    writer.close()
    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, criterion, device)

    metrics_dict = metrics(results, truths)
    print(metrics_dict)
    print("-"*50)

    return metrics_dict

def test_case(hyp_params, test_loader):
    model = getattr(Models, hyp_params.model)(hyp_params)
    model = load_model(hyp_params, name=hyp_params.name)
    if hyp_params.use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    criterion = getattr(nn, hyp_params.criterion)()

    model.return_embedding = hyp_params.return_embedding
    model.eval()
    model.to(device)
    if hyp_params.return_embedding:
        embeddings = []
        truths = []
        with torch.no_grad():
            for labels, strs, toks in test_loader:
                toks.to(device)
                y = torch.tensor(labels, device=device).long()
                batch_size = hyp_params.batch_size
                embedding = model(strs, toks)
                embeddings.append(embedding)
                truths.append(y)
        embeddings = torch.cat(embeddings).cpu().numpy()
        truths = torch.cat(truths).cpu().numpy()
        color_dict = {'Non-depolymerase': plt.cm.Set2.colors[7], 'Depolymerase': "#FFB871"}
        output_dir = hyp_params.data_path
        # plot_umap(embeddings, truths, color_dict, title="UMAP projection of hidden embedding", output_dir = output_dir, ignore_ylabel=False)
        plot_tsne(embeddings, truths, color_dict, title="T-SNE projection of hidden embedding", output_dir = output_dir, ignore_ylabel=False)
        avg_loss = 0
    else:
        total_loss = 0.0
        results = []
        truths = []
        with torch.no_grad():
            for labels, strs, toks in test_loader:
                toks.to(device)
                y = torch.tensor(labels, device=device).float().unsqueeze(-1)
                batch_size = hyp_params.batch_size
                logits = model(strs, toks)
                total_loss += criterion(logits, y).item() * batch_size

                results.append(torch.sigmoid(logits))
                truths.append(y)
        avg_loss = total_loss / hyp_params.n_test

        results = torch.cat(results)
        truths = torch.cat(truths)
        metrics_dict = metrics(results, truths)
        print('-'*50)
        print(metrics_dict)
        print('-'*50)
        preds = (results > 0.5).float()

        result_df = pd.DataFrame({'prob': results.view(-1).cpu().detach().numpy(),
                                'pred': preds.view(-1).cpu().detach().numpy()})
        result_df.to_csv(os.path.join(hyp_params.data_path, f'{hyp_params.test_data}_result.tsv'), index=False, sep='\t')

    return avg_loss


def predict(hyp_params, test_loader):
    model = getattr(Models, hyp_params.model)(hyp_params)
    model = load_model(hyp_params, name=hyp_params.name)
    if hyp_params.use_cuda:
        device = torch.device("cuda")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    model.return_attn = hyp_params.return_attn
    model.return_embedding = hyp_params.return_embedding
    model.eval()
    
    if hyp_params.return_embedding:
        embedding_dict = {}
        with torch.no_grad():
            for labels, strs, toks in test_loader:
                toks.to(device)
                embedding = model(strs, toks)
                for i in range(len(strs)):
                    embedding_dict[labels[i]] = embedding[i].cpu().numpy()
                embedding_path = os.path.join(hyp_params.data_path, f'embedding/{hyp_params.test_data}.npz')
        os.makedirs(os.path.join(hyp_params.data_path, 'embedding'), exist_ok=True)
        print(f"Saving Depolymerase embedding in {os.path.join(hyp_params.data_path, f'embedding/{hyp_params.test_data}.npz')}")
        np.savez(embedding_path, **embedding_dict)
    else:
        results = []
        label_list = []
        with torch.no_grad():
            for labels, strs, toks in test_loader:
                toks.to(device)
                if hyp_params.return_attn:
                    attn_dict = {}
                    logits, attn = model(strs, toks)
                    attn = attn.cpu().numpy()
                    for i in range(len(strs)):
                        attn_dict[labels[i]] = (strs[i], attn[i, :, :len(strs[i]), :len(strs[i])].sum(0).mean(0))
                    attn_path = os.path.join(hyp_params.data_path, 'attn/attn_npz/')
                    os.makedirs(attn_path, exist_ok=True)
                    print(f"Saving Depolymerase attention in {os.path.join(attn_path, f'{labels[0]}.npz')}")
                    np.savez(os.path.join(attn_path, f'{labels[0]}.npz'), **attn_dict)
                    # draw_attn(os.path.join(hyp_params.data_path, 'attn/'), attn_dict)
                else:
                    logits = model(strs, toks)

                results.append(torch.sigmoid(logits))
                label_list.append(labels)
        label_list = [item for sublist in label_list for item in sublist]
        probs = torch.cat(results)
        preds = (probs > 0.5).float()
        probs = probs.view(-1).cpu().detach().numpy()
        preds = preds.view(-1).cpu().detach().numpy()
        result_df = pd.DataFrame({'label': label_list, 'prob': probs, 'pred': preds})
        result_df.to_csv(os.path.join(hyp_params.data_path, f'{hyp_params.test_data.split(".")[0]}_result.tsv'), index=False, sep='\t')

    return
