import torchvision
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pickle
from pathlib import Path


def CIFAR10(outdir="./result/train"):
    outdir = Path(outdir)
    dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True,
        transform=T.ToTensor()
    )
    # image = torch.permute(image, (2, 0, 1))
    train_dataset, eval_dataset = train_test_split(dataset, test_size=0.1)
    
    # TODO StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(train_dataset)
    # train_dataset = scaler.transform(train_dataset)
    # eval_dataset = scaler.transform(eval_dataset)

    # with open(outdir / 'scaler_CIFAR10.pkl','wb') as f:
    #     pickle.dump(scaler, f)
        
    return train_dataset, eval_dataset


def CIFAR10_test(outdir="./result/test"):
    outdir = Path(outdir)
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True,
        transform=T.ToTensor()
    )
        
    return test_dataset
