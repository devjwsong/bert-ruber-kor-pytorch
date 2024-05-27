from train_module import TrainModule
from argparse import Namespace

import argparse
import torch


def extract(args):
    # Setting the checkpoint path to load.
    print(f"Extracting the checkpont info from log {args.log_idx}...")
    ckpt_dir = f"{args.default_root_dir}/lightning_logs/version_{args.log_idx}/checkpoints"
    
    # Loading the checkpoint.
    module = TrainModule.load_from_checkpoint(f"{ckpt_dir}/{args.ckpt_file}")
    
    # Parsing essential arugments/parameters for convenience.
    model_path, max_len, pooling = module.args.model_path, module.args.max_len, module.args.pooling
    w1_size, w2_size, w3_size = module.args.w1_size, module.args.w2_size, module.args.w3_size
    
    # Loading each module.
    bert, tokenizer = module.bert, module.tokenizer
    bert.config.w1_size, bert.config.w2_size, bert.config.w3_size = w1_size, w2_size, w3_size
    config = bert.config
    
    print(config)
    print(tokenizer)
    print(bert)
    
    # Setting output directory.
    model_path = f"{model_path.split('/')[0]}-{model_path.split('/')[1]}"
    output_dir = f"{ckpt_dir}/{model_path}-ruber-{max_len}-{pooling}"
    
    # Saving pre-trained config, tokenizer, and model as forms supported by Huggingface's transformers.
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    bert.save_pretrained(output_dir)
    
    torch.save(module.mlp_net.state_dict(), f"{output_dir}/MLPNetwork.pt")

if __name__=='__main__':
    # Arguments for checkpoint extraction.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--default_root_dir', type=str, default="./", help="The default directory for logs & checkpoints.")
    parser.add_argument('--log_idx', type=int, required=True, help="The lightning log index.")
    parser.add_argument('--ckpt_file', type=str, required=True, help="The checkpoint file name to extract.")
    
    args = parser.parse_args()
    
    # Running the extraction function.
    extract(args)
    
    print("FINISHED.")