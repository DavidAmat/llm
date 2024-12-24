"""
Changes:
- Save the model not in ONXX format
- Only save the trainable parameters
"""
import gc
import os
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from architectures.fine_tune_clsify_head import TransformerModule
from config import TrainConfig
from data import LexGlueDataModule


def training_loop(config: dataclass) -> TransformerModule:
    """Train and checkpoint the model with highest F1; log that model and return it."""
    model = TransformerModule(
        pretrained_model=config.pretrained_model,
        num_classes=config.num_classes,
        lr=config.lr,
    )
    datamodule = LexGlueDataModule(
        pretrained_model=config.pretrained_model,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        debug_mode_sample=config.debug_mode_sample,
    )

    # Use PyTorch Lightning CSVLogger.
    log_dir = "logging"
    os.makedirs(log_dir, exist_ok=True)

    # Create a name for the csv logger
    csv_logger_name = f"{config.pretrained_model}-lex-glue-tos"
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name=csv_logger_name
    )

    # Keep the model with the highest F1 score.
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{Val_F1_Score:.2f}",
        monitor="Val_F1_Score",
        mode="max",
        verbose=True,
        save_top_k=1,
    )

    # Run the training loop.
    trainer = Trainer(
        callbacks=[
            EarlyStopping(
                monitor="Val_F1_Score",
                min_delta=config.min_delta,
                patience=config.patience,
                verbose=True,
                mode="max",
            ),
            checkpoint_callback,
        ],
        default_root_dir=config.model_checkpoint_dir,
        fast_dev_run=bool(config.debug_mode_sample),
        max_epochs=config.max_epochs,
        max_time=config.max_time,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=csv_logger,
    )
    trainer.fit(model=model, datamodule=datamodule)
    best_model_path = checkpoint_callback.best_model_path

    # Evaluate the last and the best models on the test sample.
    trainer.test(model=model, datamodule=datamodule)
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=best_model_path,
    )

    # Save only LoRA adapter weights
    lora_save_path = os.path.join(config.model_checkpoint_dir, "lora_adapters.pt")
    torch.save(
        {name: param.cpu() for name, param in model.model.named_parameters() if param.requires_grad},
        lora_save_path
    )
    print(f"LoRA adapters saved to {lora_save_path}")

    return model, datamodule


if __name__ == "__main__":
    # Free up GPU VRAM from memory leaks.
    torch.cuda.empty_cache()
    gc.collect()

    train_config = TrainConfig()

    # Train model.
    trained_model, data_module = training_loop(train_config)
