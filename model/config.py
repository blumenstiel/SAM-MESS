from detectron2.config import CfgNode as CN


def add_sam_config(cfg):
    """
    Add config for SAM.
    """

    cfg.PROMPT_TYPE = "points"
    cfg.IGNORE_BACKGROUND = False  # ignoring background reduced performance in our experiments
    cfg.MAX_INPUTS = 100  # maximal number of inputs points/boxes
