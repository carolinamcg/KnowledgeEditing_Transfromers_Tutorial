# Knowledge Neurons in PreTrained Transformers

This code is an implementation of the orignal paper "Knowledge Neurons in PreTrained Transformers" adapted from its github repo (links below). 

https://arxiv.org/abs/2104.08696 (paper)
https://github.com/Hunter-DDM/knowledge-neurons (repo)




It focus on the identification of specific knowledge neurons (KN) in the transformer's FFN that can be altered to edit and update knowledge facts.

## Data (data/PARAREL)
Original dataset files used in the paper. *data_all_bags.json" is already organized in relational facts, each cointaing diverse input prompts to feed to the model.

## Code (scr/)

*   custom_bert.py code a transformer model with a Masked Language Modeling (MLM) head, following BERT's architecture. It's used also to load pre-trained BERT model (e.g: *bert-base-cased*).
*   KN_class.py has all core classes and methods to compute the attribution scores and save these in *rlt.jsonl* files; use these saved attribution scores to identify and extract KN, saving them as *json* files; exploit the found KN for model editting.
*   utils.py has some plotting and side functions.
*   notebook.ipynb builds the pipeline with all the functions above to showcase the implementation and analysis of this method. This is the one you want to run first!

## Results (results/)
Where the files mentioned before, along with some plots, are saved.

# Create Conda env
conda env create -f env.yml
conda activate seminar

Load the pre-trained BERT model for your evaluation. To run the *notebook.py* as it is, load the *bert-base-cased*.

    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz"
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz"
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz"
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz"
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz"
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz"
