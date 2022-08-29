# Implementation-of-JoSEC-COLING-22

Implementation of our COLING22 paper Debiasing Word Embeddings with Nonlinear Geometry [1]

Our code is adapted from [Debiasing Multiclass Word Embeddings](https://github.com/TManzini/DebiasMulticlassWordEmbedding) (NAACL 2019)

The repository has two main components. 
1. Identifying bias subspaces, performing hard-debiasing and MAC score calculations (./Debiasing/debias.py)
2. Downstream evaluations (./Downstream/BiasEvalPipelineRunner.ipynb)

## Data

1. We used pretrained Baseline Word2Vec embeddings which is available here [w2v_0](https://drive.google.com/file/d/1IJdGfnKNaBLHP9hk0Ns7kReQwo_jR1xx/view?usp=sharing)
2. We produced Word2Vecs which have been debiased using hard debiasing for [gender](https://drive.google.com/file/d/163_WFhbQTd2JcOBPxFP6LFrZlWdjj8Bf/view?usp=sharing), [race](https://drive.google.com/file/d/179DeLmMpsXsllLgS96DnZ7gCRYLhc1Ki/view?usp=sharing), [religion](https://drive.google.com/file/d/1z--9NXJV9NIoP4ZMgERvD7239wJh9Nhw/view?usp=sharing), and [intersection](https://drive.google.com/file/d/1WukMsXjmJF5UmP_xZVfy4qMyPOXbepz1/view?usp=sharing) - All based on w2v_0. 

## Debiasing
Running debias.py requires the following command line arguments
* -embeddingPath : The path to the word2vec embeddings (Defaults to w2v_0)
* vocabPath : The path to the social stereotype vocabulary
* subs : The models which produce bias subspaces ('josec' is our proposed model)
* eval : The evaluation set to calculate MAC scores  
* -hard : If this flag is set hard debiasing will be performed
* -soft : If this flag is used then soft debiasing will be performed
* -w : If this flag is used then all the output of the analogy tasks, the debiased Word2Vecs and the MAC statistics will be written to disk in an folder named "./output"
* -v : If this flag is used then the debias script will execute in verbose mode
* -k : An integer which denotes how many principal components that are used to define the bias subspace (Defaults to 2)
* -g : If this flag is used then the distribution of biased and debiased words are plotted in 2-dimensional space

Example command is included below.

This commmand performs intersectional hard debiasing based on attributes in the input vocab file. 
JoSEC is used to identify the intersectional subspace and the first 2 PCA components are used for computing the individual bias subspaces.
```
python debias.py inter josec inter -hard -v -k 2
```

## Downstream Evaluations
1. We evaluated the performance of debiased word embeddings on three tasks - POS tagging, POS chunking, and NER.
To reproduce our results, run the ipython notebook ./Downstream/BiasEvalPipelineRunner.ipynb
2. We trained simple LSTM-based Toxicity Detection model and measured FNED/FPED score on 3 demographic groups - gender/race/religion.
Datasets can be downloaded from [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification).
To reproduce our results, train the model with ./Downstream/ToxicityDetectionLSTM.ipynb and run ./Downstream/ToxicityDetectionEval.ipynb

## Requirements
The following python packages are required (Python 3).
* numpy 1.20.3
* pandas 1.3.1
* scipy 1.6.2
* gensim 3.8.3
* sklearn 0.24.2
* pytorch 1.9.0
* matplotlib 3.4.2
* jupyter 1.0.0

### Reference
> [1] [Lu Cheng](https://www.public.asu.edu/~lcheng35/), [Nayoung Kim](https://nayoungkim94.github.io/) and [Huan Liu](https://www.public.asu.edu/~huanliu/). Debiasing Word Embeddings with Nonlinear Geometry. Proceedings of the 29th International Conference on Computational Linguistics (COLING), 2022.
