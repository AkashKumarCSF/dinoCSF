import torch
import torch.distributed as dist


class Evaluator:

    def __init__(
        self,
        classifier,
        feature_extractor,
        criterion,
        device,
        rank
    ):

        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.criterion = criterion

        self.device = device
        self.rank = rank

    @torch.no_grad()
    def evaluate(self, loader):

        self.classifier.eval()

        correct = torch.tensor(
            0.0,
            device=self.device
        )

        total = torch.tensor(
            0.0,
            device=self.device
        )

        loss_sum = torch.tensor(
            0.0,
            device=self.device
        )

        for imgs, labels in loader:

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

            preds = outputs.argmax(1)

            correct += (preds == labels).sum()

            total += labels.size(0)

            loss_sum += loss.item()

        dist.all_reduce(correct)
        dist.all_reduce(total)
        dist.all_reduce(loss_sum)

        acc = correct.item() / total.item()

        avg_loss = (
            loss_sum.item()
            / dist.get_world_size()
        )

        return acc, avg_loss

    @torch.no_grad()
    def evaluate_test_metrics(
        self,
        loader
    ):

        self.classifier.eval()

        local_preds = []
        local_labels = []

        correct = torch.tensor(
            0.0,
            device=self.device
        )

        total = torch.tensor(
            0.0,
            device=self.device
        )

        loss_sum = torch.tensor(
            0.0,
            device=self.device
        )

        for imgs, labels in loader:

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

            preds = outputs.argmax(1)

            correct += (preds == labels).sum()

            total += labels.size(0)

            loss_sum += loss.item()

            local_preds.append(preds)

            local_labels.append(labels)

        dist.all_reduce(correct)
        dist.all_reduce(total)
        dist.all_reduce(loss_sum)

        acc = correct.item() / total.item()

        avg_loss = (
            loss_sum.item()
            / dist.get_world_size()
        )

        local_preds = torch.cat(local_preds)

        local_labels = torch.cat(local_labels)

        gathered_preds = [
            torch.zeros_like(local_preds)
            for _ in range(dist.get_world_size())
        ]

        gathered_labels = [
            torch.zeros_like(local_labels)
            for _ in range(dist.get_world_size())
        ]

        dist.all_gather(
            gathered_preds,
            local_preds
        )

        dist.all_gather(
            gathered_labels,
            local_labels
        )

        if self.rank == 0:

            y_pred = torch.cat(
                gathered_preds
            ).cpu().numpy()

            y_true = torch.cat(
                gathered_labels
            ).cpu().numpy()

            return (
                acc,
                avg_loss,
                y_true,
                y_pred
            )

        return acc, avg_loss, None, None