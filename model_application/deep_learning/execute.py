from tqdm import tqdm
import time
import os
import shutil
import numpy as np
import json
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.amp import GradScaler

from .data import CustomDataset
from .model import LSTM
from .model import GRU

from model_application.evaluation_errors_utils import error

class ModelExecutor:
    def __init__(
        self,
        device: str,
        learning_rate: float,
        checkpoint_path: str,
        scaler_path: str,
        model_type: str,
        input_size: int,
        hidden_size: int = 1024,
        num_layers: int = 12,
        dropout_prob: float = 0.3,
        n_steps: int = 1,
        num_epochs: int = 100,
        use_amp: bool = False,
        batch_size: int = 16
    ):
        super(ModelExecutor, self).__init__()

        self.checkpoint_path = checkpoint_path
        self.learning_rate = learning_rate
        self.device = device

        self.batch_size = batch_size

        if model_type == "LSTM":
            self.model = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_prob=dropout_prob,
                n_steps=n_steps
            ).to(self.device)
            print(self.model)
        elif model_type == "GRU":
            self.model = GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_prob=dropout_prob,
                n_steps=n_steps
            ).to(self.device)
            print(self.model)

        self.optimizer = Adam(
            self.model.parameters(),
            self.learning_rate
        )
        self.grad_scaler = GradScaler(enabled=use_amp)
        self.criterion = nn.MSELoss()

        self.epoch = 1
        self.patience = 0
        self.num_epochs = num_epochs
        self.scaler = joblib.load(os.path.join(scaler_path, "scaler.pkl"))

    def create_loader(
        self,
        train_dataset: CustomDataset,
        validation_dataset: CustomDataset,
        test_dataset: CustomDataset
    ):
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        self.validation_loader = DataLoader(validation_dataset, shuffle=False, batch_size=1)
        self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    def train(self):
        self.model.train()
        running_loss = 0.0
        with tqdm(desc="Epoch %d - Training" % self.epoch, unit="it", total=len(self.train_loader)) as pbar:
            for i, batch_item in enumerate(self.train_loader, start=1):
                features, target = batch_item
                features = features.to(self.device)
                target = target.to(self.device)
                
                output = self.model(features)
                loss = self.criterion(output, target)
                
                this_loss = loss.item()
                running_loss += this_loss

                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                pbar.set_postfix(loss=f"{running_loss / i}")
                pbar.update()

        return running_loss / len(self.train_loader)

    def evaluate(
        self,
        type: str
    ):
        if type == "validation":
            evaluation_loader = self.validation_loader
        else:
            evaluation_loader = self.test_loader

        self.model.eval()
        outputs = []
        targets = []

        running_loss = 0.0
        with tqdm(desc="Epoch %d - Evaluation" % self.epoch, unit="it", total=len(evaluation_loader)) as pbar:
            for i, batch_item in enumerate(evaluation_loader, start=1):
                with torch.no_grad():
                    features, target = batch_item
                    features = features.to(self.device)
                    target = target.to(self.device)
                    
                    output = self.model(features)
                    loss = self.criterion(output, target)

                    this_loss = loss.item()
                    running_loss += this_loss

                outputs.append(output)
                targets.append(target)
                
                pbar.set_postfix(loss=f"{running_loss / i}")
                pbar.update()
        
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)

        mse, rmse, mae, mape = error(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        print(f"Evaluation scores: MSE - {mse}, RMSE - {rmse}, MAE - {mae}, MAPE - {mape}")

        return (
            outputs,
            targets,
            running_loss / len(evaluation_loader),
            mse,
            rmse,
            mae,
            mape
        )

    def save_checkpoint(
        self,
        epoch: int,
        patience: int,
        train_loss: float,
        validation_loss: float,
        validation_mse: float,
        validation_rmse: float,
        validation_mae: float,
        validation_mape: float,
        training_time: float
    ):
        dict_for_saving = {
            "epoch": epoch,
            "patience": patience,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_mse": validation_mse,
            "validation_rmse": validation_rmse,
            "validation_mae": validation_mae,
            "validation_mape": validation_mape,
            "training_time": training_time
        }

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth"))

    def load_checkpoint(self, file_path) -> dict:
        if not os.path.exists(file_path):
            return None
        print("Loading checkpoint from ", file_path)
        checkpoint = torch.load(file_path)
        return checkpoint

    def run(
        self,
        train_dataset: CustomDataset,
        validation_dataset: CustomDataset,
        test_dataset: CustomDataset,
        patience_threshold: int = 10
    ):
        self.create_loader(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset
        )

        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            if os.path.isfile(os.path.join(self.checkpoint_path, "best_model.pth")):
                best_checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
                best_mse = best_checkpoint["validation_mse"]
            else:
                best_mse = np.inf
            
            last_checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            self.epoch = last_checkpoint["epoch"]
            self.model.load_state_dict(last_checkpoint["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
            self.patience = last_checkpoint["patience"]
            training_time = last_checkpoint["training_time"]
            
            print(f"Resuming from epoch {self.epoch}")
        else:
            training_time = 0
            best_mse = np.inf
        
        print("Start training!")
        while True:
            start_time = time.time()
            train_loss = self.train()
            end_time = time.time()
            
            best = False

            training_time += (end_time - start_time)

            (
                _,
                _,
                validation_loss,
                validation_mse,
                validation_rmse,
                validation_mae,
                validation_mape
            ) = self.evaluate(type="validation")
            
            self.save_checkpoint(
                epoch=self.epoch,
                patience=self.patience,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_mse=validation_mse,
                validation_rmse=validation_rmse,
                validation_mae=validation_mae,
                validation_mape=validation_mape,
                training_time=training_time
            )

            if validation_mse < best_mse:
                best = True
                best_mse = validation_mse
                self.patience = 0
            else:
                self.patience += 1
                
            
            if best:
                shutil.copyfile(
                    os.path.join(self.checkpoint_path, "last_model.pth"), 
                    os.path.join(self.checkpoint_path, "best_model.pth")
                )

            if self.epoch == self.num_epochs or self.patience == patience_threshold:
                break

            self.epoch += 1
        print("Finish training!")

        print("Start testing!")
        checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_time = time.time()
        predictions, targets, _, _, _, _, _ = self.evaluate(type="test")
        end_time = time.time()

        elapsed = end_time - start_time

        predictions = self.scaler.inverse_transform(predictions.detach().cpu().numpy())
        targets = self.scaler.inverse_transform(targets.detach().cpu().numpy())
        mse, rmse, mae, mape = error(predictions, targets)

        json.dump(
            {
                "time": elapsed,
                "predictions": predictions.tolist(),
                "targets": targets.tolist(),
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "mape": mape
            },
            open(os.path.join(self.checkpoint_path, "test_results.json"), "w+"),
            ensure_ascii=False,
            indent=4
        )

        print(f"Thời gian tính toán (test dataset): {elapsed:.2f} giây ({elapsed / 60:.2f} phút)")

    def get_predictions(self, features):
        if not os.path.isfile(os.path.join(self.checkpoint_path, "best_model.pth")):
            print("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path!")

        checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            predictions = self.model(features)

        json.dump(
            {"predictions": predictions.detach().cpu().numpy().tolist()},
            open(os.path.join(self.checkpoint_path, "predictions.json"), "w+"),
            ensure_ascii=False,
            indent=4
        )
