#!/bin/bash

# Performance testing script for matrix decomposition
# Tests threads from 1-16 and matrix sizes from 64 to 4096
# Each test runs 3 times to get average execution time

# Ensure output directory exists
mkdir -p results

# Output file
RESULT_FILE="results/performance_results.txt"

# Define matrix sizes and thread counts
MATRIX_SIZES=(64 128 256 512 1024 2048 4096)
THREAD_COUNTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
RUNS=3

# Compile the program
echo "Compiling the program..."
make clean
make

# Initialize result file with header
echo "Performance Results - Matrix Decomposition" > $RESULT_FILE
echo "Format: Algorithm: Thread count, Matrix size, Time (seconds)" >> $RESULT_FILE
echo "-----------------------------------------------------------" >> $RESULT_FILE

# Function to extract LU decomposition time from program output
extract_lu_time() {
    lu_time=$(echo "$1" | grep "Time:" | head -1 | awk '{print $2}')
    echo $lu_time
}

# Function to extract Cholesky decomposition time from program output
extract_cholesky_time() {
    chol_time=$(echo "$1" | grep "Time:" | tail -1 | awk '{print $2}')
    echo $chol_time
}

# Run tests for each thread count and matrix size
for thread in "${THREAD_COUNTS[@]}"; do
    for size in "${MATRIX_SIZES[@]}"; do
        lu_total_time=0
        chol_total_time=0
        echo "Testing with $thread threads, matrix size $size..."
        
        # Run the test RUNS times
        for ((run=1; run<=RUNS; run++)); do
            echo "  Run $run of $RUNS"
            output=$(./Parallel $thread $size)
            
            # Extract times for this run
            lu_time=$(extract_lu_time "$output")
            chol_time=$(extract_cholesky_time "$output")
            
            lu_total_time=$(echo "$lu_total_time + $lu_time" | bc)
            chol_total_time=$(echo "$chol_total_time + $chol_time" | bc)
        done
        
        # Calculate average times
        lu_avg_time=$(echo "scale=6; $lu_total_time / $RUNS" | bc)
        chol_avg_time=$(echo "scale=6; $chol_total_time / $RUNS" | bc)
        
        # Write to result file
        echo "LU: Thread $thread, Matrix size $size, Time $lu_avg_time" >> $RESULT_FILE
        echo "Cholesky: Thread $thread, Matrix size $size, Time $chol_avg_time" >> $RESULT_FILE
    done
    echo "Thread $thread completed for all matrix sizes."
done

echo "All tests completed. Results saved to $RESULT_FILE"

# Calculate and append speedups for LU decomposition
echo "" >> $RESULT_FILE
echo "LU Decomposition - Speedup Relative to Single Thread" >> $RESULT_FILE
echo "Format: Matrix size, Thread count, Speedup" >> $RESULT_FILE
echo "---------------------------------------" >> $RESULT_FILE

# Extract baseline times for LU (thread count = 1)
declare -A lu_baseline_times
for size in "${MATRIX_SIZES[@]}"; do
    baseline_line=$(grep "^LU: Thread 1, Matrix size $size" $RESULT_FILE)
    baseline_time=$(echo $baseline_line | awk '{print $7}')
    lu_baseline_times[$size]=$baseline_time
}

# Calculate LU speedups for each thread count and matrix size
for size in "${MATRIX_SIZES[@]}"; do
    baseline=${lu_baseline_times[$size]}
    
    for thread in "${THREAD_COUNTS[@]}"; do
        if [ "$thread" -eq 1 ]; then
            continue  # Skip thread 1 since speedup is always 1.0
        fi
        
        # Get the time for this thread count and matrix size
        thread_line=$(grep "^LU: Thread $thread, Matrix size $size" $RESULT_FILE)
        current_time=$(echo $thread_line | awk '{print $7}')
        
        # Calculate speedup
        speedup=$(echo "scale=2; $baseline / $current_time" | bc)
        echo "Size $size, Thread $thread, Speedup $speedup" >> $RESULT_FILE
    done
done

# Calculate and append speedups for Cholesky decomposition
echo "" >> $RESULT_FILE
echo "Cholesky Decomposition - Speedup Relative to Single Thread" >> $RESULT_FILE
echo "Format: Matrix size, Thread count, Speedup" >> $RESULT_FILE
echo "---------------------------------------" >> $RESULT_FILE

# Extract baseline times for Cholesky (thread count = 1)
declare -A chol_baseline_times
for size in "${MATRIX_SIZES[@]}"; do
    baseline_line=$(grep "^Cholesky: Thread 1, Matrix size $size" $RESULT_FILE)
    baseline_time=$(echo $baseline_line | awk '{print $7}')
    chol_baseline_times[$size]=$baseline_time
}

# Calculate Cholesky speedups for each thread count and matrix size
for size in "${MATRIX_SIZES[@]}"; do
    baseline=${chol_baseline_times[$size]}
    
    for thread in "${THREAD_COUNTS[@]}"; do
        if [ "$thread" -eq 1 ]; then
            continue  # Skip thread 1 since speedup is always 1.0
        fi
        
        # Get the time for this thread count and matrix size
        thread_line=$(grep "^Cholesky: Thread $thread, Matrix size $size" $RESULT_FILE)
        current_time=$(echo $thread_line | awk '{print $7}')
        
        # Calculate speedup
        speedup=$(echo "scale=2; $baseline / $current_time" | bc)
        echo "Size $size, Thread $thread, Speedup $speedup" >> $RESULT_FILE
    done
done

# Create a summary with the best speedup for each algorithm and matrix size
echo "" >> $RESULT_FILE
echo "Summary - Best Speedups" >> $RESULT_FILE
echo "Format: Algorithm, Matrix size, Best speedup, Best thread count" >> $RESULT_FILE
echo "---------------------------------------------------------" >> $RESULT_FILE

# Best speedup for LU
for size in "${MATRIX_SIZES[@]}"; do
    best_speedup=0
    best_thread=0
    baseline=${lu_baseline_times[$size]}
    
    for thread in "${THREAD_COUNTS[@]}"; do
        if [ "$thread" -eq 1 ]; then
            continue
        fi
        
        thread_line=$(grep "^LU: Thread $thread, Matrix size $size" $RESULT_FILE)
        current_time=$(echo $thread_line | awk '{print $7}')
        
        speedup=$(echo "scale=4; $baseline / $current_time" | bc)
        
        # Check if this is better than our current best
        if (( $(echo "$speedup > $best_speedup" | bc -l) )); then
            best_speedup=$speedup
            best_thread=$thread
        fi
    done
    
    echo "LU, Size $size, Speedup $best_speedup, Thread $best_thread" >> $RESULT_FILE
done

# Best speedup for Cholesky
for size in "${MATRIX_SIZES[@]}"; do
    best_speedup=0
    best_thread=0
    baseline=${chol_baseline_times[$size]}
    
    for thread in "${THREAD_COUNTS[@]}"; do
        if [ "$thread" -eq 1 ]; then
            continue
        fi
        
        thread_line=$(grep "^Cholesky: Thread $thread, Matrix size $size" $RESULT_FILE)
        current_time=$(echo $thread_line | awk '{print $7}')
        
        speedup=$(echo "scale=4; $baseline / $current_time" | bc)
        
        # Check if this is better than our current best
        if (( $(echo "$speedup > $best_speedup" | bc -l) )); then
            best_speedup=$speedup
            best_thread=$thread
        fi
    done
    
    echo "Cholesky, Size $size, Speedup $best_speedup, Thread $best_thread" >> $RESULT_FILE
done

echo "Performance testing and analysis completed successfully!"