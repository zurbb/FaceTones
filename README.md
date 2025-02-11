# FaceTones, a Multimodal NN that matches images to voices 

**Voice-image representation that matches voice embeddings with corresponding images.**

[Uni Project Page](https://engproj.cs.huji.ac.il/page/9804) | [Video Presentation](https://youtu.be/my4OlTlVjPg) | For more details on the loss, evaluation, and training pipeline, see the [project book](https://docs.google.com/document/d/1fgbt-1N5M1hWdRPAosM356PDXYrbSswYyxSKB5OV4nM/edit?usp=sharing).

For more information, you can reach us at:
- Zur Binyamini: [zur993@gmail.com](mailto:zur993@gmail.com) | [LinkedIn](https://www.linkedin.com/in/zur-binyamini/)
- Yedidya Toberman: [yedidyat@gmail.com](mailto:yedidyat@gmail.com) | [LinkedIn](https://www.linkedin.com/in/yedidya-toberman-361b71225/)


## Quality Evaluation

### Two-Image-One-Voice Classification Test

We evaluate our model by taking two images and one voice sample, then determining which image has a higher similarity to the voice. The formula for this test is:

$$
\frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{N-1} \sum_{j \neq i} \mathbb{I} \{ \text{similarity}(p_i, v_i) > \text{similarity}(p_j, v_i) \} \right)
$$

Where:
- \(p_i\) is the model's output for sample \(i\)
- \(v_i\) is the embedding of voice \(i\)
- \(N\) is the size of the validation set

This metric helps us understand how well the model contrasts between the matching image and other images.

We achieved **88% accuracy** on this test, running on 10 batches of \(N=50\), resulting in 50*49 pairs per batch. This totals to 24,500 pairs from the validation set, which were randomly picked and include noisy data.



## Dataset
![Alt text](data_pipeline.png)
To utilize this project, you will need to download the AVSpeech dataset. The dataset and download instructions can be found on the [AVSpeech website](https://looking-to-listen.github.io/avspeech/download.html).
The csv files are too large to add to github, so make sure to download them to the paths:
data/avspeech_test.csv
data/avspeech_train.csv
From the dataset, we extracted the first frame of each video segment, and took the whole audio segment.
We saved the images as JGP and the voices as MP3 files. 
We only took one sample from each youtube ID, mainly because of storage limitations. 
Notice this has a different requirements file than the main project, data/requirements.txt, so you should probably use a virtual environment.
To run:
```bash
pip install -r data/requirements.txt
python3 data/youtube_downloader.py
```
remember to set the constant TEST_OR_TRAIN at the head of the file.

## Training
![Alt text](loss_function_diagram.png)

```bash
pip install -r requirements.txt
python3 models/training.py --limit_size=<num_samples_for_training> --validation_size=<num_samples_for_validation> --batch_size=<batch_size> --run_name=<run_name> --epochs=<num_epochs> --description="<description>"
```

Checkpoints will be saved every epoch under "trained_models/<run_name>/checkpoint_<epoch>.pth".
Additional information for TensorBoard will be saved under "runs/<run_name>".
This can be used to follow the loss, and see the average similarity of positive and negative examples.

## Evaluation
We have two main methods for evaluating our model. First, we log the average similarity between the image embedding and the voice embeddings.
We expect to get high average similarity for the positive examples, and lower average similarity in the negative examples.
It can only be as good as the voice embedder itself, so our reference values are: 0.9539 for psitive, and 0.899 for negative.
These values can are written to the logs, and can be looked up using TensorBoard.
Secondly, we created a task where we take two images, choose the voice of one of them randomly, and check which image gets a higher similarity to the voice. This gives us an idea of how well the model is learning to contrast between the matching image and other images. We wrote a script that runs this test on many samples. In each batch, all of the samples will be compared, so if we choose validation_size of 500, and batch_size of 50, we are running 25000 tests.
```bash
python3 /eval/eval_sbs_all.py --run_name=<run_name> --validation_size=<total_num_of_sample> --batch_size=<batch_size> --num_workers=<num_workers>
```

## Demo
We created a simple demonstration of the two-image-one-voice classification test, which run on Streamlit.
It can be run using the following command:
```bash
streamlit run gui/app.py
```


## References


Xvectors:
  author    = {David Snyder and
               Daniel Garcia{-}Romero and
               Alan McCree and
               Gregory Sell and
               Daniel Povey and
               Sanjeev Khudanpur},
  title     = {Spoken Language Recognition using X-vectors},
  booktitle = {Odyssey 2018},
  pages     = {105--111},
  year      = {2018},
}

SpeechBrain:

@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}

DinoV2:

@misc{oquab2023dinov2,
      title={DINOv2: Learning Robust Visual Features without Supervision}, 
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2023},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Seeing Voices and Hearing Faces: Cross-modal biometric matching. (Arsha Arsha Nagrani et al. ,2018)

Learning Transferable Visual Models From Natural Language Supervision (CLIP, Alec Radford et al., 2021)



