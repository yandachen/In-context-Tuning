# Meta-learning via Language Model In-context Tuning

This is the implementation of the paper [Meta-learning via Language Model In-context Tuning](https://arxiv.org/pdf/2110.07814.pdf) (to appear at *ACL 2022*). 
## Table of Contents
* [Overview](#overview)
* [Requirements](#requirements)
* [Code Structure](#code-structure)
* [Meta-training](#meta-training)
* [Meta-testing](#meta-testing)
* [Demo](#demo)
* [How to Cite](#citation)


## Overview
In this work, we propose meta-learning via ***in-context tuning***, which recasts the few-shot learning process of task 
adaptation and task-specific prediction as a simple sequence prediction problem, where few-shot labeled examples are 
concatenated with the target example to form the model input. In-context tuning out-performs a wide variety of baselines in terms of accuracy, including raw LM prompting, MAML and 
instruction tuning. Meanwhile, sensitivity study shows that our FSL approach of in-context tuning is significantly less 
sensitive to few-shot examples and instruction wording compared to raw LM prompting.

Given the empirical effectiveness of in-context tuning, we conjecture that the few-shot learning potential of large LMs 
(e.g., GPT-3) might be broadly underestimated, and that in-context tuning can eliminate well-known artifacts of few-shot 
LM prompting such as over-sensitivity to example ordering, example selection and instruction wording.
   
You could find more details of this work in our [paper](https://arxiv.org/pdf/2110.07814.pdf).

## Requirements

To run our code, please install all the dependency packages by using the following command:
```
pip install -r requirements.txt
```

## Code Structure
- `ict.py` includes code for in-context tuning (meta-training a model on various few-shot tasks &rarr; meta-testing few-shot
  learning on unseen tasks). 

- `data_loader.py` includes code that samples few-shot in-context examples and embeds examples with natural language templates (prompts).

- `verbalized_model.py` includes code of a classification model that classifies examples by 
predicting the most likely class verbalizer (class labels are mapped one-on-one to class verbalizers). 


## Meta-training
To meta-train a model, please run the following command:
```bash
    python ict.py [--mode meta-train] [--model_name MODEL_NAME] [--task_format {mlm,clm}] [--device DEVICE] 
    [--task2examples TASK2EXAMPLES] [--task2templates TASK2TEMPLATES] [--task2verbalizers TASK2VERBALIZERS] 
    [--num_demonstrations NUM_DEMONSTRATIONS] [--example_delimiter EXAMPLE_DELIMITER] [--allow_label_overlap] 
    [--bsz BSZ] [--lr LR] [--num_warmup_steps NUM_WARMUP_STEPS] [--num_epochs NUM_EPOCHS] [--output_dir OUTPUT_DIR]
```
Here is the information of each argument to specify:

`--mode meta-train`: indicate you want to meta-train a model.

`--model_name MODEL_NAME`: name of the model.

`--task_format {mlm,clm}`: frame verbalizer classifcation as masked language modeling or causal language modeling.

`--device DEVICE`: device used for this experiment.

`--task2examples TASK2EXAMPLES`: path to the meta-training/meta-testing data file (a dictionary of task2examples).

`--task2templates TASK2TEMPLATES`: path to the meta-training/meta-testing templates file (a dictionary of task2templates).

`--task2verbalizers TASK2VERBALIZERS`: path to the meta-training/meta-testing verbalizers file (a dictionary of task2verbalizers).

`--num_demonstrations NUM_DEMONSTRATIONS`: number of few-shot demonstrations.

`--example_delimiter EXAMPLE_DELIMITER`: delimiter used to separate consecutive examples in the input text.

`--allow_label_overlap`: whether few-shot support examples are allowed to have overlapping labels with the query example.

`--bsz BSZ`: batch size for meta-training/meta-testing.

`--lr LR`: learning rate for meta-training.

`--num_warmup_steps NUM_WARMUP_STEPS`: number of warmup steps of the learning rate scheduler.

`--num_epochs NUM_EPOCHS`: number of meta-training epochs.

`--output_dir OUTPUT_DIR`: output directory to store the meta-trained model and the meta-training log file.


## Meta-testing
To meta-test a meta-trained model, please run the following command:
```bash
    python ict.py [--mode meta-test] [--model_name MODEL_NAME] [--task_format {mlm,clm}] [--device DEVICE] 
    [--task2examples TASK2EXAMPLES] [--task2templates TASK2TEMPLATES] [--task2verbalizers TASK2VERBALIZERS] 
    [--num_demonstrations NUM_DEMONSTRATIONS] [--example_delimiter EXAMPLE_DELIMITER] [--allow_label_overlap] 
    [--bsz BSZ] [--num_prefix_selections NUM_PREFIX_SELECTIONS] [--load_model_path LOAD_MODEL_PATH]
```
Here is the information of each argument to specify:

`--mode meta-test`: indicate you want to meta-test a model.

`--model_name MODEL_NAME`: name of the model.

`--task_format {mlm,clm}`: frame verbalizer classifcation as masked language modeling or causal language modeling.

`--device DEVICE`: device used for this experiment.

`--task2examples TASK2EXAMPLES`: path to the meta-training/meta-testing data file (a dictionary of task2examples).

`--task2templates TASK2TEMPLATES`: path to the meta-training/meta-testing templates file (a dictionary of task2templates).

`--task2verbalizers TASK2VERBALIZERS`: path to the meta-training/meta-testing verbalizers file (a dictionary of task2verbalizers).

`--num_demonstrations NUM_DEMONSTRATIONS`: number of few-shot demonstrations.

`--example_delimiter EXAMPLE_DELIMITER`: delimiter used to separate consecutive examples in the input text.

`--allow_label_overlap`: whether few-shot support examples are allowed to have overlapping labels with the query example.

`--bsz BSZ`: batch size for meta-training/meta-testing.

`--num_prefix_selections NUM_PREFIX_SELECTIONS`: number of demonstration sampling for each query example (result averaged across different sampled demonstrations).

`--load_model_path LOAD_MODEL_PATH`: path to the meta-trained model to evaluate with.


## Demo
As an example, we provide a code file `src/lama.py` that can be run simply with `cd src && python lama.py` 

The code meta-trains a BERT-Base model on the training tasks of fold 0 of LAMA and meta-tests the meta-trained
model on the validation and testing tasks of fold 0 of LAMA. Please refer to our [paper](https://arxiv.org/pdf/2110.07814.pdf)
for detailed information of the experiment. 

## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at `yc3384@columbia.edu`.

## Citation

```bibtex
@article{chen2021meta,
  title={Meta-learning via language model in-context tuning},
  author={Chen, Yanda and Zhong, Ruiqi and Zha, Sheng and Karypis, George and He, He},
  journal={arXiv preprint arXiv:2110.07814},
  year={2021}
}
```