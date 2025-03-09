#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "LU_decomposition.h"
#include "LU_optimized.h"
#include "Test_LU.h"

/**
 * Run performance tests on all implementations
 */
void runAllTests(int num_threads) {
    // Print test header with timestamp
    printf("=================================================================\n");
    printf("LU Decomposition Performance Tests - mihoyoMhb\n");
    printf("Date: 2025-03-09 12:18:32 UTC\n");
    printf("Testing all implementations on positive definite matrices\n");
    printf("Parallel implementation using %d threads\n", num_threads);
    printf("=================================================================\n");
    
    // Test with different matrix sizes
    int sizes[] = {5, 50, 100, 200, 500, 1000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // For storing total speedups to calculate average
    int valid_tests = 0;
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        printf("\n==================================================\n");
        printf("Testing with positive definite matrix size %d x %d\n", n, n);
        printf("==================================================\n");
        
        // Create test matrix (positive definite only)
        double **A_posdef = allocateMatrix(n);
        positiveDefiniteMatrix(A_posdef, n);
        
        printf("\nPositive definite matrix tests:\n");
        
        // Run all LU decomposition tests (original, optimized, parallel)
        testAllLU(A_posdef, n);
        
        // Run all Cholesky decomposition tests
        testAllCholesky(A_posdef, n);
        
        // Run all LU with partial pivoting tests
        testAllPivotLU(A_posdef, n);
        
        
        // Count this as a valid test for averaging
        valid_tests++;
        
        // Free memory
        freeMatrix(A_posdef, n);
    }
    
    printf("\n==================================================\n");
    printf("PERFORMANCE SUMMARY\n");
    printf("==================================================\n");
    printf("8-thread OpenMP parallelization testing completed\n");
    printf("Test run by: mihoyoMhb on 2025-03-09 12:18:32 UTC\n");
    printf("==================================================\n");
}

int main() {
    // Seed random number generator
    srand(time(NULL));
    
    // Set number of threads to use in parallel implementations
    int num_threads = 8;
    
    // Report available threads
    printf("System has %d processors available\n", omp_get_num_procs());
    printf("Running tests with %d threads\n\n", num_threads);
    
    // Run all tests with 8 threads
    runAllTests(num_threads);
    
    return 0;
}