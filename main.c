#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "LU_decomposition.h"
#include "LU_optimized.h"
#include "Test_LU.h"

/**
 * Run optimized performance tests with different matrix sizes
 * Only tests on positive definite matrices
 */
void runAllTests() {
    // Print test header with timestamp
    printf("=================================================================\n");
    printf("LU Decomposition Performance Tests - mihoyoMhb\n");
    printf("Date: 2025-03-09 11:20:07 UTC\n");
    printf("Testing serial optimizations on positive definite matrices only\n");
    printf("=================================================================\n");
    
    // Test with different matrix sizes
    int sizes[] = {200, 400, 500, 800, 1000, 1500};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // For storing total speedups to calculate average
    double total_lu_speedup = 0.0;
    double total_chol_speedup = 0.0; 
    double total_plu_speedup = 0.0;
    double total_ldlt_speedup = 0.0;
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
        
        // Test standard LU decomposition
        double speedup_lu = testLUdecomposition(A_posdef, n);
        if (speedup_lu > 0) {
            total_lu_speedup += speedup_lu;
        }
        
        // Test Cholesky decomposition
        double speedup_chol = testCholeskyDecomposition(A_posdef, n);
        if (speedup_chol > 0) {
            total_chol_speedup += speedup_chol;
        }
        
        // Test LU with partial pivoting
        double speedup_plu = testPartialPivotingLU(A_posdef, n);
        if (speedup_plu > 0) {
            total_plu_speedup += speedup_plu;
        }
        
        // Test LDLT decomposition
        double speedup_ldlt = testLDLTDecomposition(A_posdef, n);
        if (speedup_ldlt > 0) {
            total_ldlt_speedup += speedup_ldlt;
        }
        
        // Count this as a valid test for averaging
        valid_tests++;
        
        // Free memory
        freeMatrix(A_posdef, n);
    }
    
    // Print summary of speedups
    printf("\n==================================================\n");
    printf("PERFORMANCE SUMMARY\n");
    printf("==================================================\n");
    
    if (valid_tests > 0) {
        printf("Average speedups from serial optimizations:\n");
        printf("  LU Decomposition:           %.2f times faster\n", total_lu_speedup / valid_tests);
        printf("  Cholesky Decomposition:     %.2f times faster\n", total_chol_speedup / valid_tests);
        printf("  Partial Pivoting LU:        %.2f times faster\n", total_plu_speedup / valid_tests);
        printf("  LDL^T Decomposition:        %.2f times faster\n", total_ldlt_speedup / valid_tests);
        printf("  Overall Average Speedup:    %.2f times faster\n", 
               (total_lu_speedup + total_chol_speedup + total_plu_speedup + total_ldlt_speedup) / (valid_tests * 4));
    }
}

int main() {
    // Seed random number generator
    srand(time(NULL));
    
    // Run optimized performance tests
    runAllTests();
    
    return 0;
}