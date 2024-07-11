import argparse
import os
import coloredlogs, logging

logger = logging.getLogger()
coloredlogs.install()
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from models import eval_sbs as eval_sbs
import os
from models.training import ROOT_DIR
import torch
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, help='Name of the run')
parser.add_argument('--validation_size', type=int, help='Size of the validation set')
parser.add_argument('--batch_size', type=int, help='Batch size for evaluation')
parser.add_argument('--num_workers', type=int, help='Number of workers for data loading')
args = parser.parse_args()



def main():
    epoch = 0
    run_dir = os.path.join(os.getcwd(), 'eval_results', args.run_name)
    print(f"number of checkpoints: {len(os.listdir(os.path.join(ROOT_DIR, 'trained_models', args.run_name)))}")
    if os.path.exists(run_dir):
        run_dir += '_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(run_dir, exist_ok=True)
    
    while os.path.exists(os.path.join(ROOT_DIR, 'trained_models', args.run_name, f'checkpoint_{epoch}.pth')):
        logger.info(f'Evaluating checkpoint {epoch}')
        checkpoint_path = os.path.join(ROOT_DIR, 'trained_models', args.run_name, f'checkpoint_{epoch}.pth')
        results_path = os.path.join(run_dir, f'results_{epoch}.txt')
        eval_args = argparse.Namespace(model_checkpoint=checkpoint_path ,validation_size=args.validation_size, batch_size=args.batch_size,result_file_path=results_path, num_workers=args.num_workers)
        eval_sbs.main(eval_args)
        epoch += 1

    
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()