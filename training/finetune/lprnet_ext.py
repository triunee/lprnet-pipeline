"""kwon-evan LPRNet을 고치지 않고 실험 옵션을 얹는 확장.

- norm="sample": global context를 샘플마다 따로 정규화한다.
  원본은 torch.mean(f_pow)로 배치 전체를 평균 내서, 같은 이미지라도
  배치 구성(학습 256장 vs 서빙 1장)에 따라 출력이 달라진다.
- sched="cosine": epoch 단위 warmup + cosine. 원본은 T_0=10 step 단위
  warm restart라 300 epoch 동안 LR이 다시 올라갔다 내려가기를 반복한다.
- val-acc-b1: 검증 이미지를 한 장씩 넣어 잰 정확도(서빙 조건)를 함께 기록한다.

hparams에 norm/sched/warmup_epochs/max_epochs가 저장되므로,
체크포인트는 load_from_checkpoint로 같은 구조가 복원된다.

_LPRNetSampleNorm.forward는 kwon-evan/LPRNet(Apache-2.0)의 _LPRNet.forward를
옮겨 와 f_mean 한 줄만 바꾼 것이다.
"""
import torch
import torch.nn as nn

from lprnet import LPRNet
from lprnet.lprnet import _LPRNet
from lprnet.utils import accuracy


class _LPRNetSampleNorm(_LPRNet):
    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(5, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow, dim=(1, 2, 3), keepdim=True)  # 샘플별 평균
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        return torch.mean(x, dim=2)


class LPRNetExp(LPRNet):
    def __init__(self, args=None):
        super().__init__(args)
        if self.hparams.get("norm", "batch") == "sample":
            # 파라미터 이름이 같아서 원본 가중치(lprnet_kor.pt)를 그대로 불러올 수 있다
            self.LPRNet = _LPRNetSampleNorm(
                class_num=len(self.hparams.chars), dropout_rate=self.hparams.dropout_rate
            )

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)  # val-loss, val-acc (배치 단위)
        imgs, labels, lengths = batch
        logits = torch.cat([self(imgs[i : i + 1]) for i in range(imgs.shape[0])])
        acc = accuracy(logits, labels, lengths, self.hparams.chars)
        self.log("val-acc-b1", acc, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        if self.hparams.get("sched", "orig") == "orig":
            return super().configure_optimizers()

        optimizer = torch.optim.Adam(
            [
                {"params": self.STNet.parameters(), "weight_decay": self.hparams.weight_decay},
                {"params": self.LPRNet.parameters()},
            ],
            lr=self.hparams.lr,
        )
        warmup = self.hparams.warmup_epochs
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.max_epochs - warmup, eta_min=self.hparams.lr * 0.01
                ),
            ],
            milestones=[warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "name": "lr"},
        }
