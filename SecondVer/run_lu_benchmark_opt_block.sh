#!/bin/bash

# Compile the optimized code with optimization flags
gcc -o situ_parallel_opt_block situ_parallel_opt_block.c -O3 -march=native -fopenmp -ffast-math -ftree-vectorize -lm

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Problem sizes to test
SIZES=(64 128 512 1024 2048 4096)

# Thread counts to test
THREADS=(1 2 4 8 16)

# Output file
OUTPUT_FILE="lu_benchmark_results_opt_block.csv"

# Write header to output file
echo "Size,Threads,Time,Speedup" > $OUTPUT_FILE

# Function to extract execution time from program output
get_execution_time() {
    # Extract the time from the output (assuming it contains "time: X.XXX seconds")
    echo "$1" | grep "time" | awk '{print $(NF-1)}'
}

# Run the benchmarks
for size in "${SIZES[@]}"; do
    echo "Testing block-optimized version with problem size: $size"
    echo "------------------------------------------------"
    
    # Variable to store single thread average time for speedup calculations
    single_thread_avg_time=0
    
    # Run all thread counts 3 times each and take the average
    for thread in "${THREADS[@]}"; do
        echo "  Running with $thread threads..."
        
        # Variables for calculating average
        total_time=0
        
        # Run 3 times
        for run in {1..3}; do
            output=$(./situ_parallel_opt_block $thread $size)
            time_taken=$(get_execution_time "$output")
            total_time=$(echo "$total_time + $time_taken" | bc -l)
        done
        
        # Calculate average time
        avg_time=$(echo "scale=6; $total_time / 3" | bc -l)
        echo "  Average time with $thread threads: $avg_time seconds"
        
        # Store single thread time for speedup calculations
        if [ "$thread" -eq "1" ]; then
            single_thread_avg_time=$avg_time
            echo "  Baseline single thread time: $single_thread_avg_time seconds"
            echo "  Speedup with $thread threads: 1.00x"
        else
            # Calculate speedup
            speedup=$(echo "scale=4; $single_thread_avg_time / $avg_time" | bc -l)
            echo "  Speedup with $thread threads: ${speedup}x"
        fi
        
        # Record the result
        if [ "$thread" -eq "1" ]; then
            echo "$size,$thread,$avg_time,1.0" >> $OUTPUT_FILE
        else
            echo "$size,$thread,$avg_time,$speedup" >> $OUTPUT_FILE
        fi
    done
    
    echo "------------------------------------------------"
    echo ""  # Add an empty line between different problem sizes
done

echo "Benchmark completed. Results saved to $OUTPUT_FILE"