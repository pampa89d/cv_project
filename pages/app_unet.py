### IMPORT ###
import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from pytorch_lightning import LightningModule

# Для более сложных аугментаций рекомендуется использовать библиотеку Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

### Load checkpoint ###
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        # определение encoder
        self.downs = nn.ModuleList()
        for f in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, f, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1),
                nn.ReLU(inplace=True),
            ))
            in_channels = f
        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # decoder
        self.ups = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))
            self.ups.append(nn.Sequential(
                nn.Conv2d(f*2, f, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1),
                nn.ReLU(inplace=True),
            ))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[-(idx//2 + 1)]
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)
        return self.final_conv(x)


class UNetLitModule(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_channels, out_channels)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy(task="binary")
        self.val_acc   = Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.loss_fn(logits, masks)
        preds  = (torch.sigmoid(logits) > 0.5).long()
        acc    = self.train_acc(preds, masks.long()) if stage=="train" else self.val_acc(preds, masks.long())
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc,   prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# путь к чекпойнту
ckpt_path = "../models/unet_9_epoch.ckpt"

# воссоздаём архитектуру (должна совпадать с той, что использовалась при обучении)
base_model = UNet(in_channels=3, out_channels=1)  # 1 класс: лес/фон[1]

# восстанавливаем веса
lit_model: LightningModule = UNetLitModule.load_from_checkpoint(
    ckpt_path, 
    model=base_model,       # передаём базовую сеть
    lr=1e-3                 # остальные гиперпараметры
)

lit_model.eval()
lit_model.freeze()          # фиксируем параметры для чистого инференса
lit_model.to(DEVICE)

def get_val_transform(size: int = 256):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

### Predict results ###
@torch.no_grad()
def overlay_prediction_only(img_path: str,
                            model: torch.nn.Module,
                            transform=None,
                            threshold: float = 0.5,
                            alpha: float = 0.4,
                            figsize: tuple = (6, 6),
                            dpi: int = 100):
    """
    Отображает одно тестовое изображение с наложенной предсказанной моделью маской,
    когда истинная маска отсутствует.
    
    Параметры:
    - img_path: путь к RGB-изображению
    - model: LitModule или nn.Module, возвращающий логиты B×1×H×W
    - transform: функция/Albumentations.Transform, приводящая изображение к тензору C×H×W
    - threshold: порог бинаризации вероятностей
    - alpha: прозрачность слоя предсказанной маски
    - figsize: размер фигуры в дюймах
    - dpi: разрешение фигуры
    """
    # Загрузка и подготовка изображения
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    inp = transform(image=img)["image"].unsqueeze(0).to(next(model.parameters()).device)  # B×C×h×w
    
    # Предсказание вероятностей и бинаризация
    logits = model(inp)                       # B×1×h×w
    probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()  # h×w
    pred_mask = (probs > threshold).astype(float)      # h×w

    # Усреднённая уверенность по всем пикселям
    avg_conf = probs.mean() * 100  # в процентах
    
    # Приведение маски к размеру оригинала, если трансформ масштабировал
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
    pred_mask = np.array(pred_mask_img) / 255.0        # H×W
    
    # Подготовка фонового изображения
    base_img = img / 255.0
    
    # Создание RGBA-слоя для предсказанной маски (зелёный)
    overlay = np.zeros((H, W, 4))
    overlay[..., 2] = pred_mask     # зелёный канал
    overlay[..., 3] = alpha * pred_mask  # альфа-канал
    
    # Визуализация
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    ax[0].imshow(base_img)
    ax[1].imshow(base_img)
    ax[1].imshow(overlay)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[0].set_title("Basic image")
    ax[1].set_title(f"Image with Predicted (avg conf {avg_conf:.1f}%)")
    plt.show()

def visual_predict(path_):
    # проверка сслыка, файл или диерктория
    if os.path.isdir(path_):
        fld_path = os.walk(os.path.expanduser(path_))
        for folder, subfolders, files in fld_path:
            for name in files:
                if name.endswith('.jpg'):
                    img_path = os.path.join(folder, name)
                    overlay_prediction_only(img_path=img_path, 
                                            model=lit_model,
                                            transform=get_val_transform(),
                                            threshold=0.4,
                                            alpha=0.5,
                                            figsize=(10, 10),
                                            dpi=150)
    elif path_.startswith('http://') or path_.startswith('https://'):
        response = requests.get(path_)
        img = Image.open(BytesIO(response.content))
        img.save('tmp.jpg')  # сохраняем во временный файл
        img_path = 'tmp.jpg'
        overlay_prediction_only(img_path=img_path, 
                            model=lit_model,
                            transform=get_val_transform(),
                            threshold=0.7,
                            alpha=0.5,
                            figsize=(10, 10),
                            dpi=150)
    else:
        img_path=os.path.expanduser(path_)
        overlay_prediction_only(img_path=img_path, 
                                model=lit_model,
                                transform=get_val_transform(),
                                threshold=0.7,
                                alpha=0.5,
                                figsize=(10, 10),
                                dpi=150)

path_ = 'https://images.stockcake.com/public/d/2/0/d20c6b7a-67b8-450d-a133-3041d913036c_large/deforestation-versus-nature-stockcake.jpg'
visual_predict(path_)