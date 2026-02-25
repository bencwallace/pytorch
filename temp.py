import torch

# cudnn requirements:
# targets.dim == 1
# log_probs.dtype == float
# targets.dtype == int
# targets.contiguous == true
# log_probs.device == cuda
# log_probs.dim == 3

# _use_cudnn_ctc_loss:
# input_lengths: list[int]
# target_lengths: list[int]
# targets.device == cpu
# input_lengths == [log_probs.size(0) repeated]
# target_lengths[i] <= min(254, input_lengths[i])

# _use_cudnn_ctc_loss_tensor:
# input_lengths: tensor[int]
# target_lengths: tensor[int]
# targets.device == cuda
# additional requirements based on graph capture...

def call_ctc_cudnn_lists(log_probs, targets, input_lengths, target_lengths, zero_infinity):
    print(f"cudnn_lists, zero_infinity={zero_infinity}")
    assert torch._use_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0)
    log_probs_copy = log_probs.detach().requires_grad_()
    res = torch.nn.functional.ctc_loss(log_probs_copy, targets, input_lengths, target_lengths, reduction='sum', zero_infinity=zero_infinity)
    res.backward()
    print(res)
    print(log_probs_copy.grad)

def call_ctc_cudnn_tensors(log_probs, targets, input_lengths, target_lengths, zero_infinity):
    print(f"cudnn_tensors, zero_infinity={zero_infinity}")
    assert torch.ops.aten._use_cudnn_ctc_loss.Tensor(log_probs, targets, input_lengths, target_lengths, blank=0)
    log_probs_copy = log_probs.detach().requires_grad_()
    res = torch.nn.functional.ctc_loss(log_probs_copy, targets, input_lengths, target_lengths, reduction='sum', zero_infinity=zero_infinity)
    res.backward()
    print(res)
    print(log_probs_copy.grad)

def call_ctc_native(log_probs, targets, input_lengths, target_lengths, zero_infinity):
    print(f"native, zero_infinity={zero_infinity}")
    with torch.backends.cudnn.flags(enabled=False):
        log_probs_copy = log_probs.detach().requires_grad_()
        res = torch.nn.functional.ctc_loss(log_probs_copy, targets, input_lengths, target_lengths, reduction='sum', zero_infinity=zero_infinity)
        res.backward()
        print(res)
        print(log_probs_copy.grad)

def test_CTCLoss_zero_infinity_cudnn():
    target_lengths = torch.tensor([2], dtype=torch.int)
    input_lengths = torch.tensor([2], dtype=torch.int)
    targets = torch.ones(2, dtype=torch.int, device='cpu')  # 2 consecutive 1s require a blank between them
    # log_probs = torch.randn(2, 1, 3, dtype=torch.float, device='cuda').log_softmax(2).requires_grad_()
    log_probs = torch.zeros(2, 1, 3, dtype=torch.float, device='cuda').softmax(2)

    print(log_probs)

    call_ctc_native(log_probs, targets, input_lengths.tolist(), target_lengths.tolist(), zero_infinity=False)
    call_ctc_native(log_probs, targets, input_lengths.tolist(), target_lengths.tolist(), zero_infinity=True)

    print()

    call_ctc_cudnn_lists(log_probs, targets, input_lengths.tolist(), target_lengths.tolist(), zero_infinity=False)
    call_ctc_cudnn_lists(log_probs, targets, input_lengths.tolist(), target_lengths.tolist(), zero_infinity=True)

    print()

    call_ctc_cudnn_tensors(log_probs, targets, input_lengths, target_lengths, zero_infinity=False)
    call_ctc_cudnn_tensors(log_probs, targets, input_lengths, target_lengths, zero_infinity=True)

if __name__ == '__main__':
    test_CTCLoss_zero_infinity_cudnn()
