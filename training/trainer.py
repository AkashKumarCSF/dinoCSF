import torch
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        classifier,
        feature_extractor,
        train_loader,
        train_sampler,
        device,
        rank,
        lr,
        weight_decay
    ):

        self.classifier = classifier
        self.feature_extractor = feature_extractor

        self.train_loader = train_loader
        self.train_sampler = train_sampler

        self.device = device
        self.rank = rank

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train_one_epoch(self, epoch):

        self.classifier.train()

        self.train_sampler.set_epoch(epoch)

        total_loss = 0

        for imgs, labels in tqdm(
                self.train_loader,
                disable=(self.rank != 0)
        ):

            imgs = imgs.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            feats = self.feature_extractor(imgs)

            outputs = self.classifier(feats)

            loss = self.criterion(
                outputs,
                labels
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)