import torch
import time

# Use 4096x4096 matrices (approx 134M flops per multiplication)
matrix_size = 4096
duration = 5 * 60  # 5 minutes in seconds

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Prepare random matrices
a = torch.randn((matrix_size, matrix_size), device=device)
b = torch.randn((matrix_size, matrix_size), device=device)

# Start the timer
start_time = time.time()
iterations = 0

print("Starting stress test...")
while time.time() - start_time < duration:
    c = torch.mm(a, b)  # Matrix multiplication on GPU
    torch.cuda.synchronize()  # Ensure all ops complete before measuring time
    iterations += 1
    if iterations % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Iterations: {iterations}, Time: {int(elapsed)}s")

print(f"\nCompleted {iterations} iterations of {matrix_size}x{matrix_size} matrix multiplication.")
