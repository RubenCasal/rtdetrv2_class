import os
import json
import datasets


class CocoLike(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(features=datasets.Features({
            "image_id": datasets.Value("string"),
            "image": datasets.Image(),
            "objects": {
                "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("float64"), length=4)),
                "category": datasets.Sequence(datasets.ClassLabel(names=[])),
            },
        }))

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        with open(os.path.join(data_dir, "annotations/train.json")) as f:
            cats = [c["name"] for c in sorted(json.load(f)["categories"], key=lambda c: c["id"])]
        self.info.features["objects"]["category"].feature = datasets.ClassLabel(names=cats)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                "ann": os.path.join(data_dir, "annotations/train.json"),
                "img_dir": os.path.join(data_dir, "images/train"),
            }),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                "ann": os.path.join(data_dir, "annotations/val.json"),
                "img_dir": os.path.join(data_dir, "images/val"),
            }),
        ]

    def _generate_examples(self, ann, img_dir):
        with open(ann) as f:
            coco = json.load(f)
        imgs = {im["id"]: im for im in coco["images"]}
        cat_id2idx = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}
        anns_by_img = {}
        for a in coco["annotations"]:
            if a.get("iscrowd", 0) == 1:
                continue
            anns_by_img.setdefault(a["image_id"], []).append(a)
        for img_id, im in imgs.items():
            path = os.path.join(img_dir, im["file_name"])
            anns = anns_by_img.get(img_id, [])
            yield str(img_id), {
                "image_id": str(img_id),
                "image": path,
                "objects": {
                    "bbox": [a["bbox"] for a in anns],
                    "category": [cat_id2idx[a["category_id"]] for a in anns],
                },
            }
