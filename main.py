



from rtdetrv2_class import RTDETRV2_CLASS
import os 


os.environ["HF_TOKEN"] = "YOUR_HUGGING_FACE_API_KEY"



MODE = "train"
CHECKPOINT = "PekingU/rtdetr_v2_r50vd"
DATASET_PATH = "./dataset_prueba2"
OUTPUT_DIR = "prueba_entrenamiento"
IMAGE_SIZE = 288
TRAINING_EPOCHS = 40
WARMUP_STEPS = 300
LEARNING_RATE = 5e-4
BATCH_SIZE = 1


trainer = RTDETRV2_CLASS(
    mode=MODE,
    checkpoint=CHECKPOINT,
    dataset_path=DATASET_PATH,
    output_dir=OUTPUT_DIR,
    image_size=IMAGE_SIZE,
    training_epochs=TRAINING_EPOCHS,
    learning_rate=LEARNING_RATE,
    warm_up_steps=WARMUP_STEPS,
    batch_size=BATCH_SIZE

    )

trainer.train_model()











