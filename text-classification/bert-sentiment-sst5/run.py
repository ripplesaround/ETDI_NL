#!/usr/bin/env python3

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-c",
    "--bert-config",
    default="bert-large-uncased",
    help="Pretrained BERT configuration",
)
@click.option("-b", "--binary", is_flag=True, help="Use binary labels, ignore neutrals")
@click.option("-r", "--root", is_flag=True, help="Use only root nodes of SST")
@click.option(
    "-s", "--save", is_flag=True, help="Save the model files after every epoch"
)
@click.option(
    "-e", "--epoch",  default=30,help="epoch"
)
@click.option(
    "-nt", "--noise_type",  default="symmetric",help="symmetric or pairflip"
)
@click.option(
    "-nr", "--noise_rate",  default=0.5,help="噪声率"
)
@click.option(
    "-m", "--method",  default="proposed",help="方法"
)
@click.option(
    "-ba", "--batch_size",  default=64,help="batch_size"
)
def main(bert_config, binary, root, save, epoch, noise_type, noise_rate, method, batch_size):
    """Train BERT sentiment classifier."""
    if method == "baseline":
        from bert_sentiment.train import train

        print(binary,root,bert_config,save,epoch)
        # noise_rate = 0.3
        # noise_type = "symmetric"
        forget_rate = noise_rate
        noise_para = [noise_type, noise_rate]
        train(binary=binary, root=root, bert=bert_config, save=save,epochs=epoch, noise = noise_para, forget_rate = forget_rate)
    elif method == "coteaching":
        from bert_sentiment.train_coteaching import train

        print(binary,root,bert_config,save,epoch)
        # noise_rate = 0.3
        # noise_type = "symmetric"
        forget_rate = noise_rate
        noise_para = [noise_type, noise_rate]
        train(binary=binary, root=root, bert=bert_config, save=save,epochs=epoch, noise = noise_para, forget_rate = forget_rate)
    elif method == "proposed":
        from bert_sentiment.train_proposed import train

        print(binary,root,bert_config,save,epoch)
        # noise_rate = 0.3
        # noise_type = "symmetric"
        forget_rate = noise_rate
        noise_para = [noise_type, noise_rate]
        # batch_size = 32
        train(binary=binary, root=root, bert=bert_config, save=save,epochs=epoch, noise = noise_para, forget_rate = forget_rate, batch_size=batch_size)
    
    else:
        print("method 有问题")
        return


if __name__ == "__main__":
    main()
