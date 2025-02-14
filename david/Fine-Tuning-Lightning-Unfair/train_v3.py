import gc
import os
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
import mlflow
import torch
import torch.onnx
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from architectures.fine_tune_clsify_head import TransformerModule
from config import TrainConfig
from data import LexGlueDataModule

def training_loop(config: dataclass) -> TransformerModule:
    """Train and checkpoint the model with highest F1; log that model to MLflow and
    return it."""
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

    # Set up local MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    with mlflow.start_run(
        run_name=config.mlflow_run_name,
        description=config.mlflow_description,
    ) as run:
        # Configure MLFlowLogger for PyTorch Lightning
        mlf_logger = MLFlowLogger(
            experiment_name=config.mlflow_experiment_name,
            tracking_uri="http://localhost:5000",
            log_model=True,
        )
        mlf_logger._run_id = run.info.run_id
        
        # Log all non-MLflow config parameters
        mlflow.log_params(
            {k: v for k, v in asdict(config).items() if not k.startswith("mlflow_")}
        )

        # Keep the model with the highest F1 score
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{Val_F1_Score:.2f}",
            monitor="Val_F1_Score",
            mode="max",
            verbose=True,
            save_top_k=1,
        )

        # Run the training loop
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
            logger=mlf_logger,
        )
        
        trainer.fit(model=model, datamodule=datamodule)
        best_model_path = checkpoint_callback.best_model_path

        # Log the best model path to MLflow
        mlflow.log_artifact(best_model_path)
        
        # Evaluate the last and the best models on the test sample
        trainer.test(model=model, datamodule=datamodule)
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=best_model_path,
        )

    return model, datamodule

def convert_to_onnx(
    model: torch.nn.Module,
    save_path: PathLike | str,
    sequence_length: int,
    vocab_size: int,
) -> None:
    model.eval()
    dummy_input_ids = torch.randint(
        0,
        vocab_size,
        (1, sequence_length),
        dtype=torch.long,
    )
    dummy_attention_mask = torch.ones(
        (1, sequence_length),
        dtype=torch.long,
    )
    dummy_label = torch.zeros(
        1,
        dtype=torch.long,
    )
    torch.onnx.export(
        model=model,
        args=(dummy_input_ids, dummy_attention_mask, dummy_label),
        f=save_path,
        input_names=["input_ids", "attention_mask", "label"],
    )

if __name__ == "__main__":
    # Free up gpu vRAM from memory leaks
    torch.cuda.empty_cache()
    gc.collect()
    
    train_config = TrainConfig()
    
    # Train model
    trained_model, data_module = training_loop(train_config)
    
    # Save model to ONNX format
    onnx_path = os.path.join(
        train_config.model_checkpoint_dir,
        f"{train_config.mlflow_run_name}_model.onnx.pb",
    )
    
    os.makedirs(train_config.model_checkpoint_dir, exist_ok=True)
    convert_to_onnx(
        model=trained_model,
        save_path=onnx_path,
        sequence_length=train_config.max_length,
        vocab_size=data_module.tokenizer.vocab_size,
    )
    
    # Log the ONNX model to MLflow
    # with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
    #     mlflow.log_artifact(onnx_path)