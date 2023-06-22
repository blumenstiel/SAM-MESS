# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Code based on the MaskFormer Training Script.
"""
import os
import detectron2.utils.comm as comm

from functools import partial
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.utils.logger import setup_logger

from model import sam_model
from model.visual_prompts import generate_clicks, generate_boxes
from model.config import add_sam_config

import mess.datasets
from mess.evaluation.sem_seg_evaluation import MESSSemSegEvaluator



class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        assert evaluator_type == "sem_seg", f"Got {evaluator_type} for {dataset_name}. Only sem_seg is supported."
        evaluator = MESSSemSegEvaluator(
                        dataset_name,
                        distributed=True,
                        output_dir=output_folder,
                    )
        return evaluator


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):

        # add ignore label to generate_clicks
        ignore_labels = [MetadataCatalog.get(dataset_name).ignore_label]
        if cfg.IGNORE_BACKGROUND:
            # Add the background class to the ignore labels to avoid prompts for the background
            meta = MetadataCatalog.get(dataset_name)
            if hasattr(meta, 'background_class'):
                background_class = meta.background_class
            elif 'background' in meta.stuff_classes:
                background_class = meta.stuff_classes.index('background')
            elif 'Background' in meta.stuff_classes:
                background_class = meta.stuff_classes.index('Background')
            elif 'others' in meta.stuff_classes:
                background_class = meta.stuff_classes.index('others')
            elif 'Others' in meta.stuff_classes:
                background_class = meta.stuff_classes.index('Others')
            else:
                background_class = 0
            ignore_labels.append(background_class)

        # add generate_clicks to test loader
        if cfg.PROMPT_TYPE == 'points':
            collate_fn = partial(generate_clicks, ignore_labels=ignore_labels, max_inputs=cfg.MAX_INPUTS)
        elif cfg.PROMPT_TYPE == 'boxes':
            collate_fn = partial(generate_boxes, ignore_labels=ignore_labels, max_inputs=cfg.MAX_INPUTS)
        else:
            raise ValueError(f"Unknown prompt type {cfg.PROMPT_TYPE}")

        return build_detection_test_loader(cfg, dataset_name, collate_fn=collate_fn)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sam_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)

        return res
    else:
        print("Only evaluation is supported")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
