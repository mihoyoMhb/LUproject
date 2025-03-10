import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_speedup_comparison(file1, file2, label1, label2, title):
    # Load data
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Extract unique problem sizes
    sizes = sorted(df1["Size"].unique())
    colors = plt.cm.tab10.colors  # Use distinct colors
    
    plt.figure(figsize=(10, 6))
    
    for i, size in enumerate(sizes):
        subset1 = df1[df1["Size"] == size]
        subset2 = df2[df2["Size"] == size]
        
        plt.plot(
            subset1["Threads"], subset1["Speedup"], marker="o", linestyle="-", color=colors[i % len(colors)], label=f"{label1} Size {size}"
        )
        plt.plot(
            subset2["Threads"], subset2["Speedup"], marker="s", linestyle="--", color=colors[i % len(colors)], label=f"{label2} Size {size}"
        )
    
    # Ideal speedup line
    ideal_threads = np.array(sorted(df1["Threads"].unique()))
    ideal_speedup = ideal_threads
    plt.plot(ideal_threads, ideal_speedup, "k--", linewidth=1.5, label="Ideal Speedup")
    
    # Labels and title
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    
    # Show plot
    plt.show()

# File paths (replace with actual file paths on your machine)
file_naive = ".\lu_benchmark_results.csv"
file_optimized = ".\lu_benchmark_results_opt.csv"
file_block_optimized = ".\lu_benchmark_results_opt_block.csv"

# Plot Naive vs Optimized
plot_speedup_comparison(file_naive, file_optimized, "Naive", "Optimized", "Comparison of Naive vs. Optimized Speedup")

# Plot Optimized vs Block-Optimized
plot_speedup_comparison(file_optimized, file_block_optimized, "Optimized", "Blocked Optimized", "Comparison of Optimized vs. Block-Optimized Speedup")
