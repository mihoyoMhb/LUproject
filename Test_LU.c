#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "LU_decomposition.h"
#include "LU_optimized.h"
#include "LU_parallel.h"
#include "Test_LU.h"

// Allocate a dynamic 2D matrix
double** allocateMatrix(int n) {
    double **matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

// Free a dynamic 2D matrix
void freeMatrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Create a random matrix
void randomMatrix(double **A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)rand() / RAND_MAX * 20.0 - 10.0; // Values between -10 and 10
        }
    }
}

// Create a symmetric matrix
void symmetricMatrix(double **A, int n) {
    randomMatrix(A, n);
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            A[j][i] = A[i][j];
        }
    }
}

// Create a symmetric positive definite matrix
void positiveDefiniteMatrix(double **A, int n) {
    double **B = allocateMatrix(n);
    randomMatrix(B, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += B[i][k] * B[j][k];
            }
            A[i][j] = sum;
        }
        A[i][i] += n; // Add to diagonal to ensure positive definiteness
    }
    
    freeMatrix(B, n);
}

// Multiply two matrices C = A*B
void multiplyMatrices(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Calculate Frobenius norm of difference between two matrices
double matrixError(double **A, double **B, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }
    return sqrt(sum);
}

// General LU testing function for all three implementations
void testAllLU(double **A, int n, int num_threads) {
    printf("\n--- Testing LU decomposition (all implementations) ---\n");
    
    double **L_orig = allocateMatrix(n);
    double **U_orig = allocateMatrix(n);
    double **L_opt = allocateMatrix(n);
    double **U_opt = allocateMatrix(n);
    double **L_par = allocateMatrix(n);
    double **U_par = allocateMatrix(n);
    double **Result = allocateMatrix(n);
    
    // Test original implementation
    clock_t start = clock();
    LUdecomposition(A, L_orig, U_orig, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    multiplyMatrices(L_orig, U_orig, Result, n);
    double error_orig = matrixError(A, Result, n);
    
    printf("Original implementation:\n");
    printf("  Time: %f seconds\n", time_orig);
    printf("  Error: %e\n", error_orig);
    
    // Test optimized implementation
    start = clock();
    LUdecomposition_optimized(A, L_opt, U_opt, n);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC;
    
    multiplyMatrices(L_opt, U_opt, Result, n);
    double error_opt = matrixError(A, Result, n);
    double speedup_opt = time_orig / time_opt;
    
    printf("Optimized implementation:\n");
    printf("  Time: %f seconds\n", time_opt);
    printf("  Error: %e\n", error_opt);
    printf("  Speedup vs original: %.2f times\n", speedup_opt);
    
    // Test parallel implementation
    start = clock();
    LUdecomposition_parallel(A, L_par, U_par, n, num_threads);
    end = clock();
    double time_par = (double)(end - start) / CLOCKS_PER_SEC;
    
    multiplyMatrices(L_par, U_par, Result, n);
    double error_par = matrixError(A, Result, n);
    double speedup_par = time_orig / time_par;
    
    printf("Parallel implementation (%d threads):\n", num_threads);
    printf("  Time: %f seconds\n", time_par);
    printf("  Error: %e\n", error_par);
    printf("  Speedup vs original: %.2f times\n", speedup_par);
    printf("  Parallel efficiency: %.2f%%\n", (speedup_par / num_threads) * 100.0);
    
    // Display small matrices if requested
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("L matrix (original):\n");
        printMatrix(L_orig, n, n);
        printf("U matrix (original):\n");
        printMatrix(U_orig, n, n);
    }
    
    // Free memory
    freeMatrix(L_orig, n);
    freeMatrix(U_orig, n);
    freeMatrix(L_opt, n);
    freeMatrix(U_opt, n);
    freeMatrix(L_par, n);
    freeMatrix(U_par, n);
    freeMatrix(Result, n);
}

// General Cholesky testing function for all three implementations
void testAllCholesky(double **A, int n, int num_threads) {
    printf("\n--- Testing Cholesky decomposition (all implementations) ---\n");
    
    double **L_orig = allocateMatrix(n);
    double **L_opt = allocateMatrix(n);
    double **L_par = allocateMatrix(n);
    double **Result = allocateMatrix(n);
    
    // Test original implementation
    clock_t start = clock();
    int result_orig = CholeskyDecomposition(A, L_orig, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result_orig == -1) {
        printf("Original implementation: Matrix is not positive definite\n");
        freeMatrix(L_orig, n);
        freeMatrix(L_opt, n);
        freeMatrix(L_par, n);
        freeMatrix(Result, n);
        return;
    }
    
    multiplyLowerTriangular(L_orig, Result, n);
    double error_orig = matrixError(A, Result, n);
    
    printf("Original implementation:\n");
    printf("  Time: %f seconds\n", time_orig);
    printf("  Error: %e\n", error_orig);
    
    // Test optimized implementation
    start = clock();
    int result_opt = CholeskyDecomposition_optimized(A, L_opt, n);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result_opt == -1) {
        printf("Optimized implementation: Matrix is not positive definite\n");
        freeMatrix(L_orig, n);
        freeMatrix(L_opt, n);
        freeMatrix(L_par, n);
        freeMatrix(Result, n);
        return;
    }
    
    multiplyLowerTriangular(L_opt, Result, n);
    double error_opt = matrixError(A, Result, n);
    double speedup_opt = time_orig / time_opt;
    
    printf("Optimized implementation:\n");
    printf("  Time: %f seconds\n", time_opt);
    printf("  Error: %e\n", error_opt);
    printf("  Speedup vs original: %.2f times\n", speedup_opt);
    
    // Test parallel implementation
    start = clock();
    int result_par = CholeskyDecomposition_parallel(A, L_par, n, num_threads);
    end = clock();
    double time_par = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result_par == -1) {
        printf("Parallel implementation: Matrix is not positive definite\n");
        freeMatrix(L_orig, n);
        freeMatrix(L_opt, n);
        freeMatrix(L_par, n);
        freeMatrix(Result, n);
        return;
    }
    
    multiplyLowerTriangular(L_par, Result, n);
    double error_par = matrixError(A, Result, n);
    double speedup_par = time_orig / time_par;
    
    printf("Parallel implementation (%d threads):\n", num_threads);
    printf("  Time: %f seconds\n", time_par);
    printf("  Error: %e\n", error_par);
    printf("  Speedup vs original: %.2f times\n", speedup_par);
    printf("  Parallel efficiency: %.2f%%\n", (speedup_par / num_threads) * 100.0);
    
    // Display small matrices if requested
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("L matrix (original):\n");
        printMatrix(L_orig, n, n);
        printf("L matrix (optimized):\n");
        printMatrix(L_opt, n, n);
        printf("L matrix (parallel):\n");
        printMatrix(L_par, n, n);
    }
    
    // Free memory
    freeMatrix(L_orig, n);
    freeMatrix(L_opt, n);
    freeMatrix(L_par, n);
    freeMatrix(Result, n);
}

// General LU with partial pivoting testing function
void testAllPivotLU(double **A, int n, int num_threads) {
    printf("\n--- Testing LU decomposition with Partial Pivoting (all implementations) ---\n");
    
    double **L_orig = allocateMatrix(n);
    double **U_orig = allocateMatrix(n);
    int *P_orig = (int *)malloc(n * sizeof(int));
    double **PA_orig = allocateMatrix(n);
    
    double **L_opt = allocateMatrix(n);
    double **U_opt = allocateMatrix(n);
    int *P_opt = (int *)malloc(n * sizeof(int));
    double **PA_opt = allocateMatrix(n);
    
    double **L_par = allocateMatrix(n);
    double **U_par = allocateMatrix(n);
    int *P_par = (int *)malloc(n * sizeof(int));
    double **PA_par = allocateMatrix(n);
    
    double **Result = allocateMatrix(n);
    
    // Make copies of A for each implementation
    double **A_orig = allocateMatrix(n);
    double **A_opt = allocateMatrix(n);
    double **A_par = allocateMatrix(n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_orig[i][j] = A_opt[i][j] = A_par[i][j] = A[i][j];
        }
    }
    
    // Test original implementation
    clock_t start = clock();
    PartialPivotingLU(A_orig, L_orig, U_orig, P_orig, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Apply permutation to A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            PA_orig[i][j] = A[P_orig[i]][j];
        }
    }
    
    multiplyMatrices(L_orig, U_orig, Result, n);
    double error_orig = matrixError(PA_orig, Result, n);
    
    printf("Original implementation:\n");
    printf("  Time: %f seconds\n", time_orig);
    printf("  Error: %e\n", error_orig);
    
    // Test optimized implementation
    start = clock();
    PartialPivotingLU_optimized(A_opt, L_opt, U_opt, P_opt, n);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Apply permutation to A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            PA_opt[i][j] = A[P_opt[i]][j];
        }
    }
    
    multiplyMatrices(L_opt, U_opt, Result, n);
    double error_opt = matrixError(PA_opt, Result, n);
    double speedup_opt = time_orig / time_opt;
    
    printf("Optimized implementation:\n");
    printf("  Time: %f seconds\n", time_opt);
    printf("  Error: %e\n", error_opt);
    printf("  Speedup vs original: %.2f times\n", speedup_opt);
    
    // Test parallel implementation
    start = clock();
    PartialPivotingLU_parallel(A_par, L_par, U_par, P_par, n, num_threads);
    end = clock();
    double time_par = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Apply permutation to A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            PA_par[i][j] = A[P_par[i]][j];
        }
    }
    
    multiplyMatrices(L_par, U_par, Result, n);
    double error_par = matrixError(PA_par, Result, n);
    double speedup_par = time_orig / time_par;
    
    printf("Parallel implementation (%d threads):\n", num_threads);
    printf("  Time: %f seconds\n", time_par);
    printf("  Error: %e\n", error_par);
    printf("  Speedup vs original: %.2f times\n", speedup_par);
    printf("  Parallel efficiency: %.2f%%\n", (speedup_par / num_threads) * 100.0);
    
    // Display small matrices if requested
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("Permutation P (original): ");
        for (int i = 0; i < n; i++) printf("%d ", P_orig[i]);
        printf("\nL matrix (original):\n");
        printMatrix(L_orig, n, n);
        printf("U matrix (original):\n");
        printMatrix(U_orig, n, n);
    }
    
    // Free memory
    freeMatrix(L_orig, n);
    freeMatrix(U_orig, n);
    free(P_orig);
    freeMatrix(PA_orig, n);
    
    freeMatrix(L_opt, n);
    freeMatrix(U_opt, n);
    free(P_opt);
    freeMatrix(PA_opt, n);
    
    freeMatrix(L_par, n);
    freeMatrix(U_par, n);
    free(P_par);
    freeMatrix(PA_par, n);
    
    freeMatrix(Result, n);
    freeMatrix(A_orig, n);
    freeMatrix(A_opt, n);
    freeMatrix(A_par, n);
}

// General LDLT testing function
void testAllLDLT(double **A, int n, int num_threads) {
    printf("\n--- Testing LDL^T decomposition (all implementations) ---\n");
    
    double **L_orig = allocateMatrix(n);
    double *D_orig = (double *)malloc(n * sizeof(double));
    double **LD_orig = allocateMatrix(n);
    
    double **L_opt = allocateMatrix(n);
    double *D_opt = (double *)malloc(n * sizeof(double));
    double **LD_opt = allocateMatrix(n);
    
    double **L_par = allocateMatrix(n);
    double *D_par = (double *)malloc(n * sizeof(double));
    double **LD_par = allocateMatrix(n);
    
    double **Result = allocateMatrix(n);
    
    // Test original implementation
    clock_t start = clock();
    int result_orig = LDLTDecomposition(A, L_orig, D_orig, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result_orig == -1) {
        printf("Original implementation: Matrix is not symmetric or decomposition failed\n");
        freeMatrix(L_orig, n);
        free(D_orig);
        freeMatrix(LD_orig, n);
        freeMatrix(L_opt, n);
        free(D_opt);
        freeMatrix(LD_opt, n);
        freeMatrix(L_par, n);
        free(D_par);
        freeMatrix(LD_par, n);
        freeMatrix(Result, n);
        return;
    }
    
    // Calculate L*D
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LD_orig[i][j] = L_orig[i][j] * (j == i ? D_orig[j] : (j < i ? 1.0 : 0.0));
        }
    }
    
    // Calculate (L*D)*L^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Result[i][j] = 0.0;
            for (int k = 0; k <= (i < j ? i : j); k++) {
                Result[i][j] += LD_orig[i][k] * L_orig[j][k];
            }
        }
    }
    double error_orig = matrixError(A, Result, n);
    
    printf("Original implementation:\n");
    printf("  Time: %f seconds\n", time_orig);
    printf("  Error: %e\n", error_orig);
    
    // Test optimized implementation
    start = clock();
    int result_opt = LDLTDecomposition_optimized(A, L_opt, D_opt, n);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result_opt == -1) {
        printf("Optimized implementation: Matrix is not symmetric or decomposition failed\n");
        freeMatrix(L_orig, n);
        free(D_orig);
        freeMatrix(LD_orig, n);
        freeMatrix(L_opt, n);
        free(D_opt);
        freeMatrix(LD_opt, n);
        freeMatrix(L_par, n);
        free(D_par);
        freeMatrix(LD_par, n);
        freeMatrix(Result, n);
        return;
    }
    
    // Calculate L*D
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LD_opt[i][j] = L_opt[i][j] * (j == i ? D_opt[j] : (j < i ? 1.0 : 0.0));
        }
    }
    
    // Calculate (L*D)*L^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Result[i][j] = 0.0;
            for (int k = 0; k <= (i < j ? i : j); k++) {
                Result[i][j] += LD_opt[i][k] * L_opt[j][k];
            }
        }
    }
    double error_opt = matrixError(A, Result, n);
    double speedup_opt = time_orig / time_opt;
    
    printf("Optimized implementation:\n");
    printf("  Time: %f seconds\n", time_opt);
    printf("  Error: %e\n", error_opt);
    printf("  Speedup vs original: %.2f times\n", speedup_opt);
    
    // Test parallel implementation
    start = clock();
    int result_par = LDLTDecomposition_parallel(A, L_par, D_par, n, num_threads);
    end = clock();
    double time_par = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result_par == -1) {
        printf("Parallel implementation: Matrix is not symmetric or decomposition failed\n");
        freeMatrix(L_orig, n);
        free(D_orig);
        freeMatrix(LD_orig, n);
        freeMatrix(L_opt, n);
        free(D_opt);
        freeMatrix(LD_opt, n);
        freeMatrix(L_par, n);
        free(D_par);
        freeMatrix(LD_par, n);
        freeMatrix(Result, n);
        return;
    }
    
    // Calculate L*D
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LD_par[i][j] = L_par[i][j] * (j == i ? D_par[j] : (j < i ? 1.0 : 0.0));
        }
    }
    
    // Calculate (L*D)*L^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Result[i][j] = 0.0;
            for (int k = 0; k <= (i < j ? i : j); k++) {
                Result[i][j] += LD_par[i][k] * L_par[j][k];
            }
        }
    }
    double error_par = matrixError(A, Result, n);
    double speedup_par = time_orig / time_par;
    
    printf("Parallel implementation (%d threads):\n", num_threads);
    printf("  Time: %f seconds\n", time_par);
    printf("  Error: %e\n", error_par);
    printf("  Speedup vs original: %.2f times\n", speedup_par);
    printf("  Parallel efficiency: %.2f%%\n", (speedup_par / num_threads) * 100.0);
    
    // Display small matrices if requested
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("L matrix (original):\n");
        printMatrix(L_orig, n, n);
        printf("D diagonal (original): ");
        for (int i = 0; i < n; i++) printf("%8.4f ", D_orig[i]);
        printf("\n");
    }
    
    // Free memory
    freeMatrix(L_orig, n);
    free(D_orig);
    freeMatrix(LD_orig, n);
    
    freeMatrix(L_opt, n);
    free(D_opt);
    freeMatrix(LD_opt, n);
    
    freeMatrix(L_par, n);
    free(D_par);
    freeMatrix(LD_par, n);
    
    freeMatrix(Result, n);
}