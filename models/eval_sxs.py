from data_loader import get_train_loader
import coloredlogs, logging
import argparse
from training import ROOT_DIR
import os
import eval_lib as lib

parser = argparse.ArgumentParser()
parser.add_argument("--validation_size", type=int, default=128, help="Validation size of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size of the dataset")
parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model")
parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint file")
args = parser.parse_args()




def main():
   model =  
    

    
if __name__ == "__main__":
    main()