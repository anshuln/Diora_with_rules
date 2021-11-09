## Augmenting Unsupevised Constituency Parsing with Rules

Code release for the paper
> [**Augmenting Unsupevised Constituency Parsing with Rules**](https://aclanthology.org/2021.findings-acl.436/)
> 
> Atul Sahay*, Anshul Nasery*, Ayush Maheshwari, Ganesh Ramakrishnan, Rishabh Iyer
> 
appearing in the Findings of ACL 2021.

This repo forks the [official repo for DIORA](https://github.com/iesl/diora/) and builds on it. Follow the steps in the repo to setup dependencies.


## Evaluation
run 
`bash run_all_models.sh /path/to/data` to reproduce F1 score results.

## Rules
Our rules can be found in the directory `rulesets/`. The folder `rulesets/data_preprocessing` contains our code to prepare the datasets by augmenting them with rules for training

## Training
Follow instructions in the DIORA repo for preliminary instructions for training the model. Run the command
```
python3 diora/scripts/train.py \
          --arch mlp-shared \
          --batch_size 32 \
          --data_type nli\
          --elmo_cache_dir ~/path_to_dir\
          --emb elmo\
          --hidden_dim 400\
          --log_every_batch 500\
          --lr 3e-3\
          --normalize unit\
          --reconstruct_mode softmax\
          --save_after 1000\
          --train_filter_length 20\
          --train_path /path_to_data\
          --validation_path ~/path_to_data\
          --cuda\
          --use_reconstruction\
          --rule_based\
          --load_model_path ./checkpoints/diora-checkpoints/mlp-softmax-shared/model.pt
```
to start a training job

## Citation
If you find the code useful for your research, consider citing us - 
```
@inproceedings{sahay-etal-2021-rule,
    title = "Rule Augmented Unsupervised Constituency Parsing",
    author = "Sahay, Atul  and
      Nasery, Anshul  and
      Maheshwari, Ayush  and
      Ramakrishnan, Ganesh  and
      Iyer, Rishabh",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.436",
    doi = "10.18653/v1/2021.findings-acl.436",
    pages = "4923--4932",
}
```
