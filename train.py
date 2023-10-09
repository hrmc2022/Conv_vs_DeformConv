import argparse
from dataset.dataset_CIFAR10 import CIFAR10
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from model.Conv2d import Conv2d
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import train_loop
from utils.utils import generate_subset


def main(config):
    # Outdir
    outdir = Path(config["outdir"]) / 'train'
    os.makedirs(outdir, exist_ok=True)
    
    model_outdir = outdir / "ckpt"
    os.makedirs(model_outdir, exist_ok=True)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Model
    model_type : int = config["model_type"]
    if model_type == 1:
        print("model: Conv2d")
        model = Conv2d(in_channels=3, conv_ch=config["conv_ch"], num_classes=10, deform=False, modulity=False).to(device)
    elif model_type == 2:
        print("model: DeformConv2d")
        model = Conv2d(in_channels=3, conv_ch=config["conv_ch"], num_classes=10, deform=True, modulity=False).to(device)
    else:
        print("model: DeformConv2dv2")
        model = Conv2d(in_channels=3, conv_ch=config["conv_ch"], num_classes=10, deform=True, modulity=True).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Loss
    loss = nn.CrossEntropyLoss()
    
    # Dataloader
    train_dataset, eval_dataset = CIFAR10(outdir=outdir)
    
    # val_set, train_set = generate_subset(
    #     train_dataset, config["test_size"])
    # print(f'num train set: {len(train_set)}')
    # print(f'num eval set: {len(val_set)}')
    # train_sampler = SubsetRandomSampler(train_set)
    
    # train_loader = DataLoader(
    #     train_dataset, batch_size=config["batch_size"],
    #     shuffle=True, sampler=train_sampler)
    # val_loader = DataLoader(
    #     train_dataset, batch_size=config["batch_size"],
    #     sampler=val_set)
    torch.manual_seed(config["random_seed"])
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True)
    eval_loader = DataLoader(
        eval_dataset, batch_size=config["batch_size"],
        shuffle=False)
    print("train size: ", len(train_loader))
    print("eval size: ", len(eval_loader))
    
    data_loader = {"Training": train_loader, "Evaluation": eval_loader}
    
    losses = {"Training": [], "Evaluation": []}
    epochs = []
    
    # Train
    for epoch in range(1, config["num_epoch"]+1):
        epochs.append(epoch)
        
        epoch_losses = train_loop(
            device,
            epoch,
            data_loader,
            model,
            optimizer,
            loss
        )
        
        losses["Training"].append(epoch_losses["Training"])
        losses["Evaluation"].append(epoch_losses["Evaluation"])
        
        # plot
        fig, ax = plt.subplots()
        ax.plot(epochs, losses["Training"], color=(0,0,1), label="Training")
        ax.plot(epochs, losses["Evaluation"], color=(1,0,0), label="Evaluation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(bottom=0.0)
        plt.legend(loc="best")
        fig.savefig(outdir / "loss.png")
        plt.cla()
        plt.close()

        # save model
        if epoch % config["save_epoch_freq"] == 0:
            model_path = model_outdir / "model_{}.pth".format(epoch)
            torch.save(model.state_dict(), model_path)    


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, mode="r") as f:
        config = json.load(f)

    main(config)
