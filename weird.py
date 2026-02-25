import torch

target_lengths = [2]
input_lengths = [2]
targets = torch.ones(2, dtype=torch.int, device='cpu')

# TODO: depending on whether or not the next line executes before after the loss is computed, the gradients below change...
log_probs = torch.zeros(2, 1, 3, dtype=torch.float, device='cuda').softmax(2).requires_grad_()
print(log_probs)

args = (log_probs, targets, input_lengths, target_lengths)
res1 = torch.nn.functional.ctc_loss(*args, reduction='sum', zero_infinity=False)
# print(log_probs)

g1, = torch.autograd.grad(res1.sum(), log_probs)
print(g1)
