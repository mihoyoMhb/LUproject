#!/bin/bash

# Compile the optimized code with optimization flags
gcc -o situ_parallel_opt situ_parallel_opt.c -O3 -march=native -fopenmp -ffast-math -ftree-vectorize -lm

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
OUTPUT_FILE="lu_benchmark_results_opt.csv"

# Write header to output file
echo "Size,Threads,Time,Speedup" > $OUTPUT_FILE

# Function to extract execution time from program output
get_execution_time() {
    # Extract the time from the output (assuming it contains "time: X.XXX seconds")
    echo "$1" | grep "time" | awk '{print $(NF-1)}'
}

# Run the benchmarks
for size in "${SIZES[@]}"; do
    echo "Testing optimized version with problem size: $size"
    
    # Run single-threaded version (just once)
    echo "  Running with 1 thread..."
    output=$(./situ_parallel_opt 1 $size)
    single_thread_time=$(get_execution_time "$output")
    echo "  Single thread time: $single_thread_time seconds"
    
    # Record the single thread result
    echo "$size,1,$single_thread_time,1.0" >> $OUTPUT_FILE
    
    # Run multi-threaded versions (3 times each, take minimum)
    for thread in "${THREADS[@]:1}"; do  # Skip the first element (1 thread)
        echo "  Running with $thread threads..."
        
        # Run 3 times
        min_time=9999999  # Initialize to a large value
        for run in {1..3}; do
            output=$(./situ_parallel_opt $thread $size)
            time_taken=$(get_execution_time "$output")
            
            # Update minimum time if this run is faster
            if (( $(echo "$time_taken < $min_time" | bc -l) )); then
                min_time=$time_taken
            fi
        done
        
        # Calculate speedup
        speedup=$(echo "scale=4; $single_thread_time / $min_time" | bc)
        echo "  Best time with $thread threads: $min_time seconds (speedup: $speedup)"
        
        # Record the result
        echo "$size,$thread,$min_time,$speedup" >> $OUTPUT_FILE
    done
    
    echo ""  # Add an empty line between different problem sizes
done

echo "Benchmark completed. Results saved to $OUTPUT_FILE"