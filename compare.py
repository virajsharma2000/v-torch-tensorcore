import torch
import time
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

# -----------------------------
# Device check
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("GPU Name:", torch.cuda.get_device_name(0))

# -----------------------------
# Large matrices for matmul
# -----------------------------
N = 8192
a_fp32 = torch.randn((N, N), device=device, dtype=torch.float32)
b_fp32 = torch.randn((N, N), device=device, dtype=torch.float32)
a_fp16 = a_fp32.half()
b_fp16 = b_fp32.half()

# -----------------------------
# FP32 matmul timing
# -----------------------------
torch.cuda.synchronize()
start = time.time()
c_fp32 = torch.matmul(a_fp32, b_fp32)
torch.cuda.synchronize()
fp32_time = time.time() - start

# -----------------------------
# FP16 matmul timing (Tensor Cores)
# -----------------------------
torch.cuda.synchronize()
start = time.time()
c_fp16 = torch.matmul(a_fp16, b_fp16)
torch.cuda.synchronize()
fp16_time = time.time() - start

# -----------------------------
# Linear layer timings
# -----------------------------
x = torch.randn(64, 1024, device=device)
w = torch.randn(512, 1024, device=device)

# Without AMP
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y = torch.nn.functional.linear(x, w)
torch.cuda.synchronize()
no_amp_time = time.time() - start

# With AMP (Tensor Cores)
torch.cuda.synchronize()
start = time.time()
with torch.cuda.amp.autocast():
    for _ in range(100):
        y = torch.nn.functional.linear(x, w)
torch.cuda.synchronize()
amp_time = time.time() - start

# -----------------------------
# Print times
# -----------------------------
print(f"\nFP32 matmul: {fp32_time:.4f} s")
print(f"FP16 matmul: {fp16_time:.4f} s")
print(f"Linear without AMP: {no_amp_time:.4f} s")
print(f"Linear with AMP (Tensor Cores): {amp_time:.4f} s")

# -----------------------------
# PyTorch Profiler for FP16 matmul
# -----------------------------
print("\nProfiling FP16 matmul to confirm Tensor Core usage...")
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("FP16_matmul"):
        c_fp16 = torch.matmul(a_fp16, b_fp16)

top_ops = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
print(top_ops)

# -----------------------------
# Visualization
# -----------------------------
labels = ["FP32 Matmul", "FP16 Matmul", "Linear No AMP", "Linear AMP"]
times = [fp32_time, fp16_time, no_amp_time, amp_time]

plt.figure(figsize=(9,5))
bars = plt.bar(labels, times, color=['red','green','blue','orange'])
plt.ylabel("Time (s)")
plt.title("Tensor Core Performance Demo")

# Annotate bars with times
for bar, t in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.01, f"{t:.2f}s", ha='center', va='bottom')

# Annotate FP16 matmul with top profiler op
fp16_ops = prof.key_averages().table(sort_by="cuda_time_total", row_limit=3)
plt.text(1, times[1]*1.05, f"Top CUDA ops:\n{fp16_ops}", ha='center', va='bottom', fontsize=8)

plt.show()
plt.savefig("tensorcore_results.png")


A = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
B = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

torch.matmul(A, B) 
torch.cuda.synchronize()

start = time.time()
multiplited_tensor = torch.matmul(A, B)
torch.cuda.synchronize()
end = time.time()

normal_cuda = end - start

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.matmul(A, B)  # warm-up
torch.cuda.synchronize()

start = time.time()
multiplited_tensor = torch.matmul(A, B)
torch.cuda.synchronize()
end = time.time()

tensor_core = end - start

print(normal_cuda < tensor_core)

