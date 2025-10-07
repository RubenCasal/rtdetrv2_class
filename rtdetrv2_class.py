
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer
import albumentations as A
from utils import load_custom_dataset, collate_fn
from map_evaluator import MAPEvaluator
from custom_dataset import CustomDataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoConfig
import torch
from PIL import ImageDraw, ImageFont
import time


class RTDETRV2_CLASS:
    def __init__(self, checkpoint, mode="train", **kwargs):
        self.checkpoint = checkpoint
        self.mode = mode

        if self.mode == "train":
            self._init_training_components(**kwargs)
        elif self.mode == "inference":
            self._init_inference_components()
        else:
            raise ValueError("mode must be 'train' or 'inference'")

    def _init_training_components(self, dataset_path, output_dir, image_size, training_epochs, learning_rate, warm_up_steps, batch_size):
        

        # load dataset and categories
        self.dataset = load_custom_dataset(dataset_path)
        self.categories = self.dataset["train"].features["objects"].feature["category"].names
        self.id2label = {i: n for i, n in enumerate(self.categories)}
        self.label2id = {n: i for i, n in self.id2label.items()}

        # processor
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.checkpoint,
            do_resize=True,
            size={"width": image_size, "height": image_size},
            use_fast=True,
        )

        # augmentations
        self.train_transform = A.Compose([
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ], bbox_params=A.BboxParams(format="coco", label_fields=["category"]))

        self.val_transform = A.Compose([A.NoOp()], bbox_params=A.BboxParams(format="coco", label_fields=["category"]))

        # datasets
        self.train_dataset = CustomDataset(self.dataset["train"], self.image_processor, transform=self.train_transform)
        self.validation_dataset = CustomDataset(self.dataset["validation"], self.image_processor, transform=self.val_transform)
        self.test_dataset = CustomDataset(self.dataset["test"], self.image_processor, transform=self.val_transform)

        # evaluator
        self.eval_metrics = MAPEvaluator(self.image_processor, 0.01, self.id2label)

        # model and trainer
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.checkpoint, id2label=self.id2label, label2id=self.label2id, ignore_mismatched_sizes=True
        )
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_epochs,
            learning_rate=learning_rate,
            warmup_steps=warm_up_steps,
            per_device_train_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_map",
            greater_is_better=True,
            load_best_model_at_end=True,
            report_to="tensorboard"
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            data_collator=collate_fn,
            compute_metrics=self.eval_metrics,
        )



    def _init_inference_components(self):
        cfg = AutoConfig.from_pretrained(self.checkpoint)
        self.id2label = getattr(cfg, "id2label", None)
        self.label2id = getattr(cfg, "label2id", None)

        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.checkpoint,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.eval()
        
        self.class_colors = {
            "class_0": (0, 255, 0),     
            "class_1": (255, 255, 0), 
            "class_2": (255, 0, 0),       
        }

        self.font_size = 16
        self.line_width = 3

        if self.id2label is None or len(self.id2label) == 0:
            self.id2label = {0: "person", 1: "vehicle", 2: "boat"}
            self.label2id = {v: k for k, v in self.id2label.items()}

    def train_model(self):
        if self.mode != "train":
            raise RuntimeError("Cannot train in inference mode.")
        self.trainer.train()

    def predict(self, image, class_thresholds: dict):

        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)

     
        min_thr = float(min(class_thresholds.values()))
        res = self.image_processor.post_process_object_detection(
            outputs, threshold=min_thr, target_sizes=[image.size[::-1]]
        )[0]  # CPU tensors

        scores = res["scores"]               
        labels = res["labels"].to(torch.long) 
        boxes  = res["boxes"]                 

        if scores.numel() == 0:
            return {"scores": scores, "labels": labels, "boxes": boxes}

        num_classes = max(int(labels.max().item()) + 1, max(class_thresholds.keys()) + 1)
        thr_table = torch.empty(num_classes, dtype=torch.float32)
        thr_table.fill_(min_thr)  

        idx  = torch.tensor(list(class_thresholds.keys()), dtype=torch.long)
        vals = torch.tensor(list(class_thresholds.values()), dtype=torch.float32)
        thr_table.scatter_(0, idx, vals)

        thr_map = thr_table[labels]
        mask = scores >= thr_map

        return {"scores": scores[mask], "labels": labels[mask], "boxes": boxes[mask]}

    

    def draw_results(self, image, results):
      
        scores = results["scores"]
        labels = results["labels"]
        boxes = results["boxes"]

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", self.font_size)
        except:
            font = ImageFont.load_default()

        for score, label, box in zip(scores, labels, boxes):
            if isinstance(label, torch.Tensor):
                label = label.item()
            if isinstance(score, torch.Tensor):
                score = score.item()
            label_name = self.id2label[label]
            color = self.class_colors.get(label_name, (255, 255, 255))

            x1, y1, x2, y2 = map(float, box)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)
            text = f"{label_name} {score:.2f}"
            draw.text((x1, y1 - 12), text, fill=color, font=font)

      
        
        return image
            
        
