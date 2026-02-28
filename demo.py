import tabulate
import torch

print(f"PyTorch version: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def run_cuda(log_probs, targets, input_lengths, target_lengths, zero_infinity):
    log_probs.requires_grad_()
    with torch.backends.cudnn.flags(enabled=False):
        loss = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="none",
            zero_infinity=zero_infinity,
        )
        loss.backward()
        return loss, log_probs.grad

def run_cudnn(log_probs, targets, input_lengths, target_lengths, zero_infinity):
    targets = targets.cpu()
    input_lengths = input_lengths.tolist()
    target_lengths = target_lengths.tolist()
    assert torch._use_cudnn_ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=0,
    )
    loss, grad = torch._cudnn_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        deterministic=True,
        zero_infinity=zero_infinity,
    )
    return loss, grad

def run_cudnn_tensor(log_probs, targets, input_lengths, target_lengths, zero_infinity):
    assert torch.ops.aten._use_cudnn_ctc_loss.Tensor(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=0,
    )
    loss, grad = torch._cudnn_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        deterministic=True,
        zero_infinity=zero_infinity,
    )
    return loss, grad

BACKENDS = [
    ("cuda", run_cuda),
    ("cudnn", run_cudnn),
    ("cudnn_tensor", run_cudnn_tensor),
]
ZERO_INFINITY_VALUES = [False, True]

def is_finite(t):
    return torch.isfinite(t).all().item()

def run_all(log_probs, targets, input_lengths, target_lengths, example_name):
    print(f"\n{'='*60}")
    print(f"Example: {example_name}")
    print('='*60)

    results = {}
    for name, f in BACKENDS:
        results[name] = {}
        for zi in ZERO_INFINITY_VALUES:
            lp = log_probs.detach().clone()
            loss, grad = f(lp, targets, input_lengths, target_lengths, zi)
            results[name][zi] = (loss.detach(), grad.detach())

    col_headers = ["Backend"] + [f"zero_infinity={zi}" for zi in ZERO_INFINITY_VALUES]

    loss_rows = [
        [name] + [results[name][zi][0].item() if is_finite(results[name][zi][0]) else "inf/nan" for zi in ZERO_INFINITY_VALUES]
        for name, _ in BACKENDS
    ]
    grad_rows = [
        [name] + ["finite" if is_finite(results[name][zi][1]) else "inf/nan" for zi in ZERO_INFINITY_VALUES]
        for name, _ in BACKENDS
    ]

    print("\nLoss finiteness:")
    print(tabulate.tabulate(loss_rows, headers=col_headers, tablefmt="grid"))
    print("\nGrad finiteness:")
    print(tabulate.tabulate(grad_rows, headers=col_headers, tablefmt="grid"))

log_probs = torch.nn.functional.log_softmax(torch.randn(2, 1, 3, device=device), dim=-1)
targets = torch.tensor([0, 0], device=device, dtype=torch.int32)  # repeated symbol with no blank in between
input_lengths = torch.tensor([2], device=device, dtype=torch.int32)
target_lengths = torch.tensor([2], device=device, dtype=torch.int32)
run_all(log_probs, targets, input_lengths, target_lengths, "Impossible alignments")

N = 500
i = 0
j = (i + 1) % N
probs = torch.nn.functional.one_hot(torch.tensor([i], device=device), num_classes=500).float()
log_probs = torch.log(probs).unsqueeze(1)
targets = torch.tensor([j], device=device, dtype=torch.int32)
input_lengths = torch.tensor([1], device=device, dtype=torch.int32)
target_lengths = torch.tensor([1], device=device, dtype=torch.int32)
run_all(log_probs, targets, input_lengths, target_lengths, "Loss blow up")
