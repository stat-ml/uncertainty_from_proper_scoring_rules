from losses import BrierScoreLoss, NegLogScore, SphericalScoreLoss, targets2vector
from torch.nn import CrossEntropyLoss
import torch

def test_targets2vector():
    target = torch.LongTensor([5])
    target_vector = targets2vector(target, n_classes=10)
    assert torch.all(target_vector == torch.tensor(
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]))


def test_targets2vector_dist():
    prob_vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.])
    target_vector = targets2vector(prob_vector, n_classes=5)
    assert torch.all(target_vector == prob_vector)


def test_target2vector_batch():
    target = torch.LongTensor([9, 2])
    target_vector = targets2vector(target, n_classes=10)

    assert torch.all(target_vector == torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]))

def test_ce_loss():
    target = torch.LongTensor([3])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((1, 10))
    pred_probs = torch.softmax(predictions, dim=-1)
    
    loss = CrossEntropyLoss()
    pred = loss(predictions, target_vector)
    true_value = -torch.sum(target_vector * torch.log(pred_probs), dim=-1)
    assert torch.isclose(pred, true_value)


def test_ce_loss_batch():
    target = torch.LongTensor([3, 5, 0, 9])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((4, 10))
    pred_probs = torch.softmax(predictions, dim=-1)
    
    loss = CrossEntropyLoss()
    pred = loss(predictions, target_vector)
    true_value = -torch.mean(
            torch.sum(target_vector * torch.log(pred_probs), dim=-1)
            )
    assert torch.isclose(pred, true_value)


def test_brier_loss():
    target = torch.LongTensor([3])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((1, 10))
    pred_probs = torch.softmax(predictions, dim=-1)
    
    loss = BrierScoreLoss()
    pred = loss(predictions, target_vector)
    true_value = torch.sum(torch.pow(target_vector - pred_probs, 2), dim=-1)

    assert torch.isclose(pred, true_value)


def test_brier_loss_batch():
    target = torch.LongTensor([3, 5, 0, 9])
    target_vector = targets2vector(target, n_classes=10)

    predictions = torch.randn((4, 10))
    pred_probs = torch.softmax(predictions, dim=-1)
    
    loss = BrierScoreLoss()
    pred = loss(predictions, target_vector)
    true_value = torch.mean(
            torch.sum(torch.pow(target_vector - pred_probs, 2), dim=-1)
            )
    assert torch.isclose(pred, true_value)



if __name__ == "__main__":
    test_targets2vector()
    test_target2vector_batch()
    test_targets2vector_dist()
    test_ce_loss_batch()
    test_ce_loss()
    test_brier_loss()
    test_brier_loss_batch()
