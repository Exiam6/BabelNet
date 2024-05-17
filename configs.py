import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--save_dir', type=str, default="/scratch/zz4330/MMRec_Babel/Result_Babel", help='Directory to save results')

    parser.add_argument('--debug', default=False, help='Only runs 20 batches per epoch for debugging, and set debug to true', action='store_true')
    parser.add_argument('--row', type=int, default=30, help='ROW')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate'
    parser.add_argument('--load_from_checkpoint',  default=False,action='store_true', help='Load model from a checkpoint')
    parser.add_argument('--random_seed', type=int, default=43, help='Random seed')
    return parser.parse_args()

def main():
    args = get_args()

    print(f"Device: {args.device}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save directory: {args.save_dir}")
    print(f"Debug mode: {args.debug}")


if __name__ == "__main__":
    main()
