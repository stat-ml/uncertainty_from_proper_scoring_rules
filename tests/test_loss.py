import numpy as np
import torch
from psruq.losses import BrierScoreLoss, NegLogScore, SphericalScoreLoss
from psruq.utils import targets2vector
from torch.nn import CrossEntropyLoss


def test_targets2vector():
    target = torch.LongTensor([5])
    target_vector = targets2vector(target, n_classes=10)
    assert torch.all(target_vector == torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]))


def test_targets2vector_dist():
    prob_vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0])
    target_vector = targets2vector(prob_vector, n_classes=5)
    assert torch.all(target_vector == prob_vector)


def test_target2vector_batch():
    target = torch.LongTensor([9, 2])
    target_vector = targets2vector(target, n_classes=10)

    assert torch.all(
        target_vector
        == torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
        )
    )


def test_ce_loss():
    target = torch.LongTensor([3])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((1, 10))
    pred_probs = torch.softmax(predictions, dim=-1)

    loss = CrossEntropyLoss()
    pred_loss = loss(predictions, target_vector)
    true_value = -torch.sum(target_vector * torch.log(pred_probs), dim=-1)
    assert torch.isclose(pred_loss, true_value)


def test_ce_loss_batch():
    target = torch.LongTensor([3, 5, 0, 9])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((4, 10))
    pred_probs = torch.softmax(predictions, dim=-1)

    loss = CrossEntropyLoss()
    pred_loss = loss(predictions, target_vector)
    true_value = -torch.mean(torch.sum(target_vector * torch.log(pred_probs), dim=-1))
    assert torch.isclose(pred_loss, true_value)


def test_brier_loss():
    target = torch.LongTensor([3])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((1, 10))
    pred_probs = torch.softmax(predictions, dim=-1)

    loss = BrierScoreLoss()
    pred_loss = loss(predictions, target_vector)
    true_value = torch.sum(torch.pow(target_vector - pred_probs, 2), dim=-1)

    assert torch.isclose(pred_loss, true_value)


def test_brier_loss_batch():
    target = torch.LongTensor([3, 5, 0, 9])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((4, 10))
    pred_probs = torch.softmax(predictions, dim=-1)

    loss = BrierScoreLoss()
    pred_loss = loss(predictions, target_vector)
    true_value = torch.mean(torch.sum(torch.pow(target_vector - pred_probs, 2), dim=-1))
    assert torch.isclose(pred_loss, true_value)


def test_spherical_score_loss():
    target = torch.LongTensor([3])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((1, 10))
    pred_probs = torch.softmax(predictions, dim=-1)

    loss = SphericalScoreLoss()
    pred_loss = loss(predictions, target_vector)

    normed_pred = pred_probs / torch.norm(pred_probs, p=2, dim=-1)
    normed_target = target_vector / torch.norm(target_vector, p=2, dim=-1)

    true_value = -torch.mean(
        torch.norm(target_vector, p=2, dim=-1)
        * torch.sum(normed_pred * normed_target, dim=-1)
    )

    assert torch.isclose(pred_loss, true_value)


def test_spherical_score_loss_batch():
    target = torch.LongTensor([3, 5, 0, 9])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((4, 10))
    pred_probs = torch.softmax(predictions, dim=-1)

    loss = SphericalScoreLoss()
    pred_loss = loss(predictions, target_vector)

    normed_pred = pred_probs / torch.norm(pred_probs, dim=-1, keepdim=True, p=2)
    normed_target = target_vector / torch.norm(target_vector, dim=-1, keepdim=True, p=2)

    true_value = torch.mean(
        -torch.norm(target_vector, dim=-1, p=2)
        * torch.sum(normed_pred * normed_target, dim=-1)
    )
    assert torch.isclose(pred_loss, true_value)


def test_neglog_score_loss():
    target = torch.LongTensor([3])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((1, 10))
    pred_probs = torch.softmax(predictions, dim=-1)

    loss = NegLogScore()
    pred_loss = loss(predictions, target_vector)

    true_value = torch.sum(
        torch.log(pred_probs) - 1 + target_vector / pred_probs, dim=-1
    )

    assert torch.isclose(pred_loss, true_value)


def test_neglog_score_loss_batch():
    target = torch.LongTensor([3, 5, 0, 9])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((4, 10))
    pred_probs = torch.softmax(predictions, dim=-1)

    loss = NegLogScore()
    pred_loss = loss(predictions, target_vector)

    true_value = torch.mean(
        torch.sum(torch.log(pred_probs) - 1 + target_vector / pred_probs, dim=-1)
    )
    assert torch.isclose(pred_loss, true_value)


def test_specific_ce():
    inputs_ = torch.tensor([[0.1, 0.8, 0.1]])
    targets_ = torch.tensor([[0.2, 0.3, 0.5]])
    loss = CrossEntropyLoss()
    loss_pred = loss(inputs_, targets_).cpu().numpy().item()
    assert np.isclose(1.1797266409702165, loss_pred)


def test_specific_brier():
    inputs_ = torch.tensor([[0.1, 0.8, 0.1]])
    targets_ = torch.tensor([[0.2, 0.3, 0.5]])
    loss = BrierScoreLoss()
    loss_pred = loss(inputs_, targets_, is_logit=False).cpu().numpy().item()
    assert np.isclose(0.42000000000000004, loss_pred)


def test_specific_spherical():
    inputs_ = torch.tensor([[0.1, 0.8, 0.1]])
    targets_ = torch.tensor([[0.2, 0.3, 0.5]])
    loss = SphericalScoreLoss()
    loss_pred = loss(inputs_, targets_, is_logit=False).cpu().numpy().item()
    assert np.isclose(-0.3815836220359314, loss_pred)


def test_specific_neglog():
    inputs_ = torch.tensor([[0.1, 0.8, 0.1]])
    targets_ = torch.tensor([[0.2, 0.3, 0.5]])
    loss = NegLogScore()
    loss_pred = loss(inputs_, targets_, is_logit=False).cpu().numpy().item()
    assert np.isclose(-0.4533137373023006, loss_pred)


if __name__ == "__main__":
    test_targets2vector()
    test_target2vector_batch()
    test_targets2vector_dist()
    test_ce_loss_batch()
    test_ce_loss()
    test_brier_loss()
    test_brier_loss_batch()
    test_spherical_score_loss()
    test_spherical_score_loss_batch()
    test_neglog_score_loss()
    test_neglog_score_loss_batch()
    test_specific_ce()
    test_specific_brier()
    test_specific_spherical()
    test_specific_neglog()
