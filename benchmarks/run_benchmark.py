import os
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# App configuration
URL = "http://localhost:8000/predict"
NUM_SEQUENTIAL = 100
NUM_CONCURRENT = 200
MAX_WORKERS = 10

# Generate a sample payload matching the model's expected 28 PCA features + Amount
sample_payload = {
    f"V{i}": np.random.randn() for i in range(1, 29)
}
sample_payload["Amount"] = float(np.random.uniform(1.0, 1000.0))

def send_request():
    """Send a single post request and measure the latency."""
    start_time = time.time()
    try:
        response = requests.post(URL, json=sample_payload, timeout=5.0)
        success = response.status_code == 200
    except Exception:
        success = False
    
    latency = time.time() - start_time
    return success, latency

def run_sequential_benchmark():
    """Send sequential requests to measure baseline per request time."""
    print(f"--- Running Sequential Benchmark ({NUM_SEQUENTIAL} requests) ---")
    latencies = []
    successes = 0
    
    start_total = time.time()
    for _ in range(NUM_SEQUENTIAL):
        success, latency = send_request()
        if success:
            successes += 1
        latencies.append(latency)
        
    total_time = time.time() - start_total
    
    print(f"Total time elapsed:         {total_time:.4f}s")
    print(f"Average time per request:   {np.mean(latencies):.4f}s")
    print(f"Success rate:               {successes / NUM_SEQUENTIAL * 100:.2f}%")
    print()
    return latencies, total_time

def run_concurrent_benchmark():
    """Send concurrent requests to test system limits."""
    print(f"--- Running Concurrent Benchmark ({NUM_CONCURRENT} requests, {MAX_WORKERS} workers) ---")
    latencies = []
    success_count = 0
    
    start_total = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map out concurrent executions
        results = list(executor.map(lambda _: send_request(), range(NUM_CONCURRENT)))
        
    total_time = time.time() - start_total
    
    for success, latency in results:
        if success:
            success_count += 1
        latencies.append(latency)
        
    success_rate = (success_count / NUM_CONCURRENT) * 100
    rps = NUM_CONCURRENT / total_time
    
    # Calculate percentiles
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Total time elapsed:         {total_time:.4f}s")
    print(f"Requests per second (RPS):  {rps:.2f}")
    print(f"Success rate:               {success_rate:.2f}%")
    print(f"p50 latency (median):       {p50:.4f}s")
    print(f"p95 latency:                {p95:.4f}s")
    print(f"p99 latency:                {p99:.4f}s")
    print()
    
    return latencies

def plot_histogram(latencies, filepath):
    """Plot the latencies as a histogram."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=30, color='skyblue', edgecolor='black')
    
    # Add vertical lines for percentiles
    plt.axvline(np.percentile(latencies, 50), color='green', linestyle='dashed', linewidth=2, label='p50 (Median)')
    plt.axvline(np.percentile(latencies, 95), color='orange', linestyle='dashed', linewidth=2, label='p95')
    plt.axvline(np.percentile(latencies, 99), color='red', linestyle='dashed', linewidth=2, label='p99')
    
    plt.title('Latency Distribution (Concurrent Requests)')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    plt.savefig(filepath)
    plt.close()
    print(f"Histogram saved to {filepath}")

def main():
    print("Pre-warming the API with a single request...")
    send_request()
    print()
    
    # 1. Sequential Benchmark
    run_sequential_benchmark()
    
    # 2. Concurrent Benchmark
    concurrent_latencies = run_concurrent_benchmark()
    
    # 3. Plot Histogram
    plot_histogram(concurrent_latencies, "benchmarks/latency_histogram.png")

if __name__ == "__main__":
    main()
