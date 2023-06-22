#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam

# Benchmark
export DETECTRON2_DATASETS="datasets"
TEST_DATASETS="bdd100k_sem_seg_val dark_zurich_sem_seg_val mhp_v1_sem_seg_test foodseg103_sem_seg_test atlantis_sem_seg_test dram_sem_seg_test isaid_sem_seg_val isprs_potsdam_sem_seg_test_irrg worldfloods_sem_seg_test_irrg floodnet_sem_seg_test uavid_sem_seg_val kvasir_instrument_sem_seg_test chase_db1_sem_seg_test cryonuseg_sem_seg_test paxray_sem_seg_test_lungs paxray_sem_seg_test_bones paxray_sem_seg_test_mediastinum paxray_sem_seg_test_diaphragm corrosion_cs_sem_seg_test deepcrack_sem_seg_test pst900_sem_seg_test zerowaste_sem_seg_test suim_sem_seg_test cub_200_sem_seg_test cwfid_sem_seg_test"

# Run experiments
for DATASET in $TEST_DATASETS
do
 # Base model with oracle points
 python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS facebook/sam-vit-base OUTPUT_DIR output/SAM_base_points/$NAME$DATASET PROMPT_TYPE points
 # Large model with oracle points
 python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS facebook/sam-vit-large OUTPUT_DIR output/SAM_large_points/$NAME$DATASET PROMPT_TYPE points
 # Huge model with oracle points
 python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS facebook/sam-vit-huge OUTPUT_DIR output/SAM_huge_points/$NAME$DATASET PROMPT_TYPE points
 
 # Base model with oracle boxes
 python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS facebook/sam-vit-base OUTPUT_DIR output/SAM_base_boxes/$NAME$DATASET PROMPT_TYPE boxes
 # Large model with oracle boxes
 python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS facebook/sam-vit-large OUTPUT_DIR output/SAM_large_boxes/$NAME$DATASET PROMPT_TYPE boxes
 # Huge model with oracle boxes
 python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS facebook/sam-vit-huge OUTPUT_DIR output/SAM_huge_boxes/$NAME$DATASET PROMPT_TYPE boxes
 
done

# Combine results
python mess/evaluation/mess_evaluation.py --model_outputs output/SAM_base_points output/SAM_large_points output/SAM_huge_points output/SAM_base_boxes output/SAM_large_boxes output/SAM_huge_boxes

# Run evaluation with:
# nohup bash mess/eval.sh > eval.log &
# tail -f eval.log
