import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
import logomaker as lm
import matplotlib as mpl

def label2index(label):
    label_dict = {'neg': 0, 'pos': 1}
    index = label_dict[label]
    return index


def metrics(score, y):
    pred = (score > 0.5).float()
    score = score.view(-1).cpu().detach().numpy()
    pred = pred.view(-1).cpu().detach().numpy()
    y = y.view(-1).cpu().detach().numpy()
    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    roc_auc = roc_auc_score(y, score)
    average_precision = average_precision_score(
        y, score)

    metrics_dict = {"Accuracy": accuracy, "Precision": precision,
                    "Recall": recall, "F1-score": f1, "AUC": roc_auc, "AUPRC": average_precision}
    return metrics_dict


def save_load_name(args, name=''):
    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    os.makedirs('./model/', exist_ok=True)
    torch.save(model, f'./model/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    model = torch.load(f'./model/{name}.pt', map_location=device)
    return model

def draw_attn(output_dir, attn_dict):
    
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['legend.title_fontsize'] = 11
    mpl.rcParams['legend.fontsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    
    if not os.path.exists(os.path.join(output_dir, 'img')):
        os.mkdir(os.path.join(output_dir, 'img'))
    # if not os.path.exists(os.path.join(output_dir, 'pdf')):
    #     os.mkdir(os.path.join(output_dir, 'pdf'))
    offset = .1
    for key, value in attn_dict.items():
        seq = list(value[0])
        attn = value[1]
        attn = np.sqrt(attn)
        max_attn = max(attn)
        min_attn = min(attn)

        attn = (attn-min_attn) / (max_attn-min_attn)
        df = pd.DataFrame({'character': seq, 'value': attn})
        saliency_df = lm.saliency_to_matrix(
            seq=df['character'], values=df['value']+offset)

        rows = math.ceil(len(saliency_df) / 100)
        labels = [0.00, 0.25, 0.50, 0.75, 1.00]

        fig, axes = plt.subplots(rows, 1, figsize=(12, 2*rows))

        for i in range(rows):
            tempdf = saliency_df[i*100:(i+1)*100]
            tempdf.set_index(
                [pd.Index(np.array(tempdf.index)-100*i)], inplace=True)
            logo = lm.Logo(tempdf,
                        color_scheme='skylign_protein',
                        vpad=0,
                        width=0.8,
                        font_weight='normal',
                        ax=axes if rows == 1 else axes[i])

            logo.ax.set_ylim([0, 1])
            logo.ax.set_yticks(np.array(labels) + offset)
            logo.ax.set_yticklabels(labels)
            logo.ax.set_xlim([-1, 100])
            logo.ax.set_xticks([19, 39, 59, 79, 99])
            logo.ax.set_xticklabels(np.array([20, 40, 60, 80, 100]) + 100*i)
            logo.style_spines(visible=False)
            logo.style_spines(spines=['left'], visible=True)
            logo.ax.axhline(offset, color='gray', linewidth=1, linestyle='--')
        
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"img/{key}_attn.png"), dpi=300)
        # plt.savefig(os.path.join(output_dir, f"pdf/{key}_attn.pdf"))
        plt.close()

def plot_tsne(x, y, color_dict, title, output_dir,ignore_ylabel=False):

    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['legend.title_fontsize'] = 11
    mpl.rcParams['legend.fontsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x)
    
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 6))

    for i, (name, color) in enumerate(color_dict.items()):
        plt.scatter(x_norm[y == i, 0], x_norm[y == i, 1], c=color, alpha=0.8, s=10, label=name)
 
    plt.xlabel("T-SNE Dimension 1")
    if not ignore_ylabel:
        plt.ylabel("T-SNE Dimension 2")

    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, f"img/tsne.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"pdf/tsne.pdf"))

# def plot_umap(x, y, color_dict, title, output_dir, ignore_ylabel=False):
#     reducer = umap.UMAP(random_state=42)
#     x_umap = reducer.fit_transform(x)
    
#     x_min, x_max = x_umap.min(0), x_umap.max(0)
#     x_norm = (x_umap - x_min) / (x_max - x_min)
    
#     for i, (name, color) in enumerate(color_dict.items()):
#         plt.scatter(x_norm[y == i, 0], x_norm[y == i, 1], c=color, alpha=0.8, s=10, label=name)
        
#     plt.xlabel("UMAP Dimension 1")
#     if not ignore_ylabel:
#         plt.ylabel("UMAP Dimension 2")

#     plt.title(title)
#     plt.legend(loc="lower right")
#     plt.savefig(os.path.join(output_dir, f"img/umap.png"), dpi=300)
#     plt.savefig(os.path.join(output_dir, f"pdf/umap.pdf"))