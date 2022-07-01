from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, lr_monitor, EarlyStopping
from train_module import TrainModule
from custom_dataset import CustomDataset, PadCollate
from torch.utils.data import DataLoader

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):
    # Setting the PyTorch Lightning module for training.
    print(f"Loading training module for {args.model_path}...")
    module = TrainModule(args)
    args = module.args
    
    # Setting the datasets & dataloaders.
    print("Loading datasets & dataloaders...")
    train_set = CustomDataset(args, f"{args.data_dir}/train_dials.json", module.tokenizer)
    eval_set = CustomDataset(args, f"{args.data_dir}/eval_dials.json", module.tokenizer)
    ppd = PadCollate(args.pad_id)
    
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, collate_fn=ppd.pad_collate, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=args.eval_batch_size, collate_fn=ppd.pad_collate, num_workers=args.num_workers, pin_memory=True)
        
    # Calculating the number of total training steps & warmup steps.
    num_batches = len(train_loader)
    q, r = divmod(num_batches, len(args.gpus))
    num_batches = q if r == 0 else q+1
    args.num_training_steps = args.num_epochs * num_batches
    args.num_warmup_steps = int(args.warmup_ratio * args.num_training_steps)
    
    # Setting the callbacks (ModelCheckpoint, EarlyStopping, LRMonitor).
    print("Setting pytorch lightning callback & trainer...")
    filename = "{epoch}_{step}_{train_f1:.4f}_{valid_f1:.4f}"
    monitor="valid_f1"
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        verbose=True,
        monitor=monitor,
        mode="max",
        every_n_epochs=1,
        save_weights_only=True,
    )
    lr_callback = lr_monitor.LearningRateMonitor(
        logging_interval='step',
        log_momentum=False
    )
    stopping_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0.01,
        patience=1,
        verbose=True,
        mode='max'
    )
    
    # Setting the trainer.                                   
    seed_everything(args.seed, workers=True)
    trainer = Trainer(
        default_root_dir=args.default_root_dir,
        gpus=args.gpus,
        check_val_every_n_epoch=1,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.max_grad_norm,
        strategy="ddp",
        deterministic=True,
        callbacks=[checkpoint_callback, stopping_callback, lr_callback],
    )
    
    # Training.
    print("Train starts.")
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=eval_loader)
    print("Training done.")

    print("GOOD BYE.")
    

if __name__=="__main__":
    # Arguments for training.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--model_path', type=str, required=True, help="The BERT checkpoint to load.")
    parser.add_argument('--default_root_dir', type=str, default="./", help="The default directory for logs & checkpoints.")
    parser.add_argument('--data_dir', type=str, default="data", help="The directory which contains data files.")
    parser.add_argument('--max_len', type=int, default=512, help="The maxmium length of each sequence.")
    parser.add_argument('--num_epochs', type=int, default=5, help="The number of total epochs.")
    parser.add_argument('--train_batch_size', type=int, default=16, help="The batch size for training.")
    parser.add_argument('--eval_batch_size', type=int, default=8, help="The batch size for evaluation.")
    parser.add_argument('--num_workers', type=int, default=4, help="The number of workers for data loading.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The ratio of warmup steps to total training steps.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The max gradient for gradient clipping.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="The initial learning rate.")
    parser.add_argument('--gpus', type=str, default="0", help="The indices of GPUs to use.")
    parser.add_argument('--pooling', type=str, required=True, help="The pooling method.")
    parser.add_argument('--w1_size', type=int, default=768, help="The size of w1 embedding.")
    parser.add_argument('--w2_size', type=int, default=256, help="The size of w2 embedding.")
    parser.add_argument('--w3_size', type=int, default=64, help="The size of w3 embedding.")
    parser.add_argument('--num_hists', type=int, default=0, help="The number of extra histories.")

    args = parser.parse_args()
    
    assert args.pooling in ["cls", "mean", "max"], "Specify a correct pooling method."
    assert args.num_hists >= 0, "The number of extra contexts should be 0 or higher."
    
    # Converting the string args.gpus into a list of integers.
    args.gpus = [int(gpu.replace(',', '').strip()) for gpu in args.gpus.split(" ")]

    # Running the training function.
    run(args)
