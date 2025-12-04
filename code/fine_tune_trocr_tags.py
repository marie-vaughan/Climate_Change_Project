"""
Fine-tune TrOCR on synthetic garment-tag images (with CodeCarbon emissions tracking).
"""

import os, torch, logging
from PIL import Image
from datasets import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from codecarbon import EmissionsTracker

# ---------------------------------------------------------
# 0️⃣  Configure logging + emissions folder
# ---------------------------------------------------------
LOG_DIR = "./emissions"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "training.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1️⃣  Load synthetic dataset
# ---------------------------------------------------------
def load_synthetic(split_path):
    records = []
    for fname in os.listdir(split_path):
        if fname.endswith(".txt"):
            with open(os.path.join(split_path, fname), encoding="utf-8") as f:
                text = f.read().strip()
            img_path = os.path.join(split_path, fname.replace(".txt", ".jpg"))
            if os.path.exists(img_path):
                records.append({"image": img_path, "text": text})
    return Dataset.from_list(records)

train_ds = load_synthetic("../synthetic_tags/train")
val_ds   = load_synthetic("../synthetic_tags/val")
logger.info(f"Loaded {len(train_ds)} train, {len(val_ds)} val samples.")

# ---------------------------------------------------------
# 2️⃣  Load model + processor
# ---------------------------------------------------------
MODEL_NAME = "microsoft/trocr-base-printed"
processor  = TrOCRProcessor.from_pretrained(MODEL_NAME)
model      = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.eos_token_id           = processor.tokenizer.eos_token_id
print(model.config.decoder_start_token_id, model.config.eos_token_id)

# ---------------------------------------------------------
# 3️⃣  Preprocessing
# ---------------------------------------------------------
MAX_LEN = 128
def preprocess(batch):
    images = [Image.open(p).convert("RGB") for p in batch["image"]]
    labels = processor.tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN
    ).input_ids
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    batch["pixel_values"] = pixel_values
    batch["labels"] = torch.tensor(labels)
    return batch

train_ds = train_ds.map(preprocess, batched=True, batch_size=8)
val_ds   = val_ds.map(preprocess,   batched=True, batch_size=8)
train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
val_ds.set_format(type="torch",   columns=["pixel_values", "labels"])

# ---------------------------------------------------------
#  Training configuration
# ---------------------------------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_tag_finetune",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=1000,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor.feature_extractor,
)

# ---------------------------------------------------------
# 5️⃣  Emissions tracking
# ---------------------------------------------------------
tracker = EmissionsTracker(
    project_name="TrOCR_Tag_FineTune",
    output_dir=LOG_DIR,
    output_file="emissions.csv",
    save_to_file=True,
    log_level="info"
)

logger.info("Starting emissions tracker...")
tracker.start()

try:
    trainer.train()
finally:
    emissions: float = tracker.stop()
    logger.info(f"Training completed. Total emissions: {emissions:.4f} kg CO2eq")

# ---------------------------------------------------------
# 6️⃣  Save model + processor
# ---------------------------------------------------------
model.save_pretrained("./trocr_tag_finetune")
processor.save_pretrained("./trocr_tag_finetune")
logger.info("Model saved to ./trocr_tag_finetune")


