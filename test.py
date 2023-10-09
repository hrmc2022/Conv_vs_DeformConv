import argparse
from dataset.dataset_CIFAR10 import CIFAR10_test
import json
from model.Conv2d import Conv2d
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def main(config):
    # Outdir
    outdir = Path(config["outdir"]) / 'test'
    os.makedirs(outdir, exist_ok=True)
    
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
    
    model.load_state_dict(torch.load(config["model"]))

    # Dataloader
    test_dataset = CIFAR10_test(outdir=outdir)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    print("test size: ", len(test_loader))
    
    model.eval()
    with torch.no_grad():
        acc = 0.0
        for batch, (image, target) in enumerate(test_loader):
            image = image.to(device)
            # print(image.shape)
            target = target.to("cpu")
            #print(target.shape)
                        
            out = model(image).to("cpu")

            out_label = torch.argmax(out, dim=1)
            acc += torch.sum(target == out_label)
        
        print("Acculacy: {}".format(acc / (len(test_loader) * config["batch_size"])))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, mode="r") as f:
        config = json.load(f)

    main(config)
