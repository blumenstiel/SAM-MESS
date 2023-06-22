
import torch

from transformers import SamModel, SamProcessor
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.data import MetadataCatalog
from .visual_prompts import generate_clicks, generate_boxes


@META_ARCH_REGISTRY.register()
class SAM(torch.nn.Module):

    @configurable
    def __init__(self,
                 *,
                 model_name: str,
                 num_classes: int,
                 background_class: int = 0,
                 prompt_type: str = "points",
                 ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.num_classes = num_classes
        self.background_class = background_class
        self.prompt_type = prompt_type
        if self.background_class > self.num_classes:
            # background_class == ignore_label
            self.background_class = self.num_classes

    @classmethod
    def from_config(cls, cfg):
        meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        # Find background class
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

        return {
            "model_name": cfg.MODEL.WEIGHTS,
            "num_classes": len(meta.stuff_classes),
            "background_class": background_class,
            "prompt_type": cfg.PROMPT_TYPE,
        }

    def forward(self, batched_inputs):
        # get inputs from batch
        images = [x["image"].to(self.device) for x in batched_inputs]

        if self.prompt_type == 'points':
            if 'input_points' not in batched_inputs[0]:
                # simulate inputs if not in batch
                assert batched_inputs[0]['sem_seg'] is not None, "No input_points or sem_seg in batched_inputs"
                batched_inputs = generate_clicks(batched_inputs)

            # make sure that the input points are at the same scale as the input image
            if batched_inputs[0]['image'].shape[1:] != batched_inputs[0]['sem_seg'].shape:
                rescale_inputs(batched_inputs)

            # get points from batch
            input_points = [x["input_points"] for x in batched_inputs]
            inputs = self.processor(images, input_points=input_points, return_tensors="pt").to(self.device)

        elif self.prompt_type == 'boxes':
            if 'input_boxes' not in batched_inputs[0]:
                # simulate inputs if not in batch
                assert batched_inputs[0]['sem_seg'] is not None, "No input_boxes or sem_seg in batched_inputs"
                batched_inputs = generate_boxes(batched_inputs)

            # make sure that the input boxes are at the same scale as the input image
            if batched_inputs[0]['image'].shape[1:] != batched_inputs[0]['sem_seg'].shape:
                rescale_inputs(batched_inputs)

            # get points from batch
            input_boxes = [x["input_boxes"] for x in batched_inputs]
            inputs = self.processor(images, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        else:
            print(f"Prompt type {self.prompt_type} not implemented")
            raise NotImplementedError

        # inference
        outputs = self.model(**inputs)

        # postprocess (non max suppression and resizing)
        sizes = [(i.get("height"), i.get("width")) for i in batched_inputs]
        input_sizes = [(i.shape[1], i.shape[2]) for i in images]
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), sizes,
                                                                          input_sizes, binarize=False)
        # convert instance results to semantic results
        input_classes = [x["input_classes"] for x in batched_inputs]
        processed_results = []
        for mask, classes in zip(masks, input_classes):
            # select the first out of 3 prediction masks
            pred = mask[:, 0]

            # init prediction mask
            mask_size = pred.shape[-2:]
            r = torch.zeros((self.num_classes + 1, *mask_size))

            # convert instance mask to semantic mask by selecting the highest score for each pixel and class
            classes = torch.Tensor(classes)[:, 0].int()
            for c in torch.unique(classes):
                r[c] = torch.max(pred[classes == c, :, :], dim=0).values

            # set pixels with no prediction (all scores 0. or below) to background class
            r[self.background_class, r.max(dim=0).values <= 0] = 1.

            # drop ignore class
            processed_results.append({
                "sem_seg": r[:-1],
            })
        return processed_results


def rescale_inputs(batched_inputs):
    for i, input in enumerate(batched_inputs):
        # rescale input points from mask size to input size
        scale = input['image'].shape[2] / input['sem_seg'].shape[1]

        if 'input_points' in input:
            batched_inputs[i]['input_points'] = [
                [(int(x[0][0] * scale), int(x[0][1] * scale))]
                 for x in input['input_points']
            ]
        if 'input_boxes' in input:
            batched_inputs[i]['input_boxes'] = [
                [(int(x[0][0] * scale), int(x[0][1] * scale), int(x[0][2] * scale), int(x[0][3] * scale))]
                 for x in input['input_boxes']
            ]
    return batched_inputs
