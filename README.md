# FaceTones, a co-modal NN that matches images to voices 

This project aims to create a voice-image representation that matches voice embeddings with corresponding images.

## Dataset

To utilize this project, you will need to download the AVSpeech dataset. The dataset and download instructions can be found on the [AVSpeech website](https://looking-to-listen.github.io/avspeech/download.html).

## Running
pip install -r requirements.txt
python3 models/training.py

Checkpoints will be saved every epoch under "trained_models/<run_name>/checkpoint_<epoch>.pth".
Additional information for TensorBoard will be saved under "runs/<run_name>".
This can be used to follow the loss, and see the average similarity of positive and negative examples.

## Evaluation


## References

For more information on the speech embedding technique used in this project, please refer to the following research paper:

- [Speech Embeddings: A Comparison of Models and Training Techniques](https://arxiv.org/pdf/1705.02304) (2017)

If you use this project for your own research, please make sure to cite the paper.


