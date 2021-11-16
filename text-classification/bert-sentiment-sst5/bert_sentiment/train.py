import os

import torch
from loguru import logger
from pytorch_transformers import BertConfig, BertForSequenceClassification
from tqdm import tqdm

from .data import SSTDataset
from matplotlib import pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, logits = model(batch, labels=labels)
        err = lossfn(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_labels = torch.argmax(logits, axis=1)
        train_acc += (pred_labels == labels).sum().item()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc



def evaluate_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]
            error = lossfn(logits, labels)
            loss += error.item()
            pred_labels = torch.argmax(logits, axis=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc


def train(
    root=True,
    binary=False,
    bert="bert-large-uncased",
    epochs=30,
    batch_size=64,
    save=False,
    noise = ["symmetric", 0.5],
    forget_rate = 0.5,
    num_gradual = 5
):
    print("baseline")
    noise_type ,noise_rate = noise
    print(noise_type, noise_rate)
    trainset = SSTDataset("train", root=root, binary=binary,noise_type=noise_type,noise_rate=noise_rate)
    devset = SSTDataset("dev", root=root, binary=binary)
    testset = SSTDataset("test", root=root, binary=binary)

    config = BertConfig.from_pretrained(bert)
    if not binary:
        config.num_labels = 5
    model = BertForSequenceClassification.from_pretrained(bert, config=config)

    model = model.to(device)
    print(model)
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_acc_list = []
    test_acc_list = []

    for epoch in range(1, epochs):
        train_loss, train_acc = train_one_epoch(
            model, lossfn, optimizer, trainset, batch_size=batch_size
        )
        val_loss, val_acc = evaluate_one_epoch(
            model, lossfn, optimizer, devset, batch_size=batch_size
        )
        test_loss, test_acc = evaluate_one_epoch(
            model, lossfn, optimizer, testset, batch_size=batch_size
        )
        logger.info(f"epoch={epoch}")
        logger.info(
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}"
        )
        logger.info(
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}"
        )
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        if save:
            label = "binary" if binary else "fine"
            nodes = "root" if root else "all"
            torch.save(model, f"{bert}__{nodes}__{label}__e{epoch}.pickle")


    
    # plot
    xrange = [(i) for i in range(1, epochs)]
    plt.plot(xrange, train_acc_list, 'r', label='train')
    plt.plot(xrange, test_acc_list, 'skyblue',linestyle='dashed', label='test')
    # plt.plot(xrange, test_clean_acc2_list, 'lightgreen',linestyle='dashed', label='nr=0')
    # plt.plot(xrange, test_noisy_acc2_list, 'orange',linestyle='dashed', label='nr=1')
    plt.legend()
    plt.title('{}_{}'.format(noise_type, noise_rate))
    plt.savefig('./pic/{}_{}_{}_{}.png'.format("model",epochs,noise_type, noise_rate))
    plt.clf()

    logger.success("Done!")