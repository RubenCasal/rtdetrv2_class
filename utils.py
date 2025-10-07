from datasets import load_dataset, Dataset, DatasetDict, Features, Sequence, Value, ClassLabel, Image
import numpy as np
import torch

def load_custom_dataset(data_dir):
    src = load_dataset("./coco_like.py", data_dir=data_dir, trust_remote_code=True)
    class_names = src["train"].features["objects"]["category"].feature.names

    cppe_features = Features({
        "image_id": Value("int64"),
        "image": Image(decode=True),
        "width": Value("int32"),
        "height": Value("int32"),
        "objects": Sequence(feature={
            "id": Value("int64"),
            "area": Value("int64"),
            "bbox": Sequence(Value("float32"), length=4),
            "category": ClassLabel(names=class_names),
        }),
    })

    def split_to_list(split):
        examples = []
        for idx in range(len(split)):
            ex = split[idx]
            img = ex["image"]; w, h = img.size
            bboxes = ex["objects"]["bbox"]; cats = ex["objects"]["category"]
            objs = []
            for j, (bb, c) in enumerate(zip(bboxes, cats)):
                area = float(bb[2]) * float(bb[3])
                objs.append({
                    "id": int(j),
                    "area": int(area),
                    "bbox": [np.float32(x) for x in bb],
                    "category": int(c),
                })
            try:
                image_id_int = int(ex["image_id"])
            except Exception:
                image_id_int = int(idx)
            examples.append({
                "image_id": image_id_int,
                "image": img,
                "width": int(w),
                "height": int(h),
                "objects": objs,
            })
        return examples

    train_list = split_to_list(src["train"])
    val_list   = split_to_list(src["validation"])

    ds_out = DatasetDict({
        "train": Dataset.from_list(train_list, features=cppe_features),
        "validation": Dataset.from_list(val_list, features=cppe_features),
        "test": Dataset.from_list(val_list, features=cppe_features),
    })

    print(ds_out)
    print(ds_out["train"].features)

    # --------- Forma 1 (directa, vía Features) ----------
    categories = ds_out["train"].features["objects"].feature["category"].names
    id2label = {i: n for i, n in enumerate(categories)}
    label2id = {n: i for i, n in id2label.items()}

    print("\n[Forma 1] categories =", categories)
    print("[Forma 1] id2label   =", id2label)
    print("[Forma 1] label2id   =", label2id)

    return ds_out

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data

