# Multi-domain Evaluation of Semantic Segmentation (MESS) with SAM (Segment Anything Model)

[[Website (soon)](https://github.io)] [[arXiv (soon)](https://arxiv.org/)] [[GitHub](https://github.com/blumenstiel/MESS)]

This directory contains the code for the MESS evaluation of SAM.

## Model

SAM is developed for instance segmentation. 
In order to perform semantic segmentation, multiple visual prompts are provided, specifically, one prompt for each connected segment. 
The prompts are oracle points or oracle bounding boxes (see [model/visual_prompts.py](model/visual_prompts.py) for details). 
Next, the model combines all predicted segments of each class to perform semantic segmentation. 
The oracle results are upper bounds for potential combinations of SAM with other models that provide the visual prompts (e.g., open-vocabulary object detection).   

## Setup
Create a conda environment `sam` and install the required packages. See [mess/README.md](mess/README.md) for details.
```sh
 bash mess/setup_env.sh
```

Prepare the datasets by following the instructions in [mess/DATASETS.md](mess/DATASETS.md). The `sam` env can be used for the dataset preparation. If you evaluate multiple models with MESS, you can change the `dataset_dir` argument and the `DETECTRON2_DATASETS` environment variable to a common directory (see [mess/DATASETS.md](mess/DATASETS.md) and [mess/eval.sh](mess/eval.sh), e.g., `../mess_datasets`). 

The SAM weights are downloaded automaticly by the transformers package.

## Evaluation
To evaluate the SAM models on the MESS datasets, run
```sh
bash mess/eval.sh

# for evaluation in the background:
nohup bash mess/eval.sh > eval.log &
tail -f eval.log 
```

For evaluating a single dataset, select the DATASET from [mess/DATASETS.md](mess/DATASETS.md), the DETECTRON2_DATASETS path, the PROMPT_TYPE (`points` or `boxes`), and run
```
conda activate sam
export DETECTRON2_DATASETS="datasets"
DATASET=<dataset_name>

# Base model with oracle boxes
python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS facebook/sam-vit-base OUTPUT_DIR output/SAM_base_boxes/$NAME$DATASET PROMPT_TYPE boxes
# Large model with oracle boxes
python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS facebook/sam-vit-large OUTPUT_DIR output/SAM_large_boxes/$NAME$DATASET PROMPT_TYPE boxes
# Huge model with oracle boxes
python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS facebook/sam-vit-huge OUTPUT_DIR output/SAM_huge_boxes/$NAME$DATASET PROMPT_TYPE boxes
```
