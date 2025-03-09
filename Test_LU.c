#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "LU_decomposition.h"
#include "LU_optimized.h"
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
    // First create a random matrix
    double **B = allocateMatrix(n);
    randomMatrix(B, n);
    
    // Then form A = B*B^T + n*I to ensure it's positive definite
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

// Test standard LU decomposition (original and optimized)
double testLUdecomposition(double **A, int n) {
    printf("\n--- Testing LU decomposition ---\n");
    
    // For original algorithm
    double **L = allocateMatrix(n);
    double **U = allocateMatrix(n);
    double **LU = allocateMatrix(n);
    
    // Measure time for original algorithm
    clock_t start = clock();
    LUdecomposition(A, L, U, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Verify result: A = L*U
    multiplyMatrices(L, U, LU, n);
    double error_orig = matrixError(A, LU, n);
    
    printf("Original algorithm:\n");
    printf("  Time taken: %f seconds\n", time_orig);
    printf("  Reconstruction error: %e\n", error_orig);
    
    // For optimized algorithm
    double **L_opt = allocateMatrix(n);
    double **U_opt = allocateMatrix(n);
    double **LU_opt = allocateMatrix(n);
    
    // Measure time for optimized algorithm
    start = clock();
    LUdecomposition_optimized(A, L_opt, U_opt, n);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Verify result: A = L*U
    multiplyMatrices(L_opt, U_opt, LU_opt, n);
    double error_opt = matrixError(A, LU_opt, n);
    
    double speedup = time_orig / time_opt;
    
    printf("Optimized algorithm:\n");
    printf("  Time taken: %f seconds\n", time_opt);
    printf("  Reconstruction error: %e\n", error_opt);
    printf("  Speedup: %.2f times\n", speedup);
    
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("L matrix (original):\n");
        printMatrix(L, n, n);
        printf("U matrix (original):\n");
        printMatrix(U, n, n);
        printf("L matrix (optimized):\n");
        printMatrix(L_opt, n, n);
        printf("U matrix (optimized):\n");
        printMatrix(U_opt, n, n);
    }
    
    freeMatrix(L, n);
    freeMatrix(U, n);
    freeMatrix(LU, n);
    freeMatrix(L_opt, n);
    freeMatrix(U_opt, n);
    freeMatrix(LU_opt, n);
    
    return speedup;
}

// Test Cholesky decomposition (original and optimized)
double testCholeskyDecomposition(double **A, int n) {
    printf("\n--- Testing Cholesky decomposition ---\n");
    
    // For original algorithm
    double **L = allocateMatrix(n);
    double **LLT = allocateMatrix(n);
    
    // Measure time for original algorithm
    clock_t start = clock();
    int result = CholeskyDecomposition(A, L, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result == -1) {
        printf("Original algorithm: Cholesky decomposition failed - Matrix is not positive definite\n");
        freeMatrix(L, n);
        freeMatrix(LLT, n);
        return -1.0;
    }
    
    // Verify result: A = L*L^T
    multiplyLowerTriangular(L, LLT, n);
    double error_orig = matrixError(A, LLT, n);
    
    printf("Original algorithm:\n");
    printf("  Time taken: %f seconds\n", time_orig);
    printf("  Reconstruction error: %e\n", error_orig);
    
    // For optimized algorithm
    double **L_opt = allocateMatrix(n);
    double **LLT_opt = allocateMatrix(n);
    
    // Measure time for optimized algorithm
    start = clock();
    int result_opt = CholeskyDecomposition_optimized(A, L_opt, n);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result_opt == -1) {
        printf("Optimized algorithm: Cholesky decomposition failed - Matrix is not positive definite\n");
        freeMatrix(L, n);
        freeMatrix(LLT, n);
        freeMatrix(L_opt, n);
        freeMatrix(LLT_opt, n);
        return -1.0;
    }
    
    // Verify result: A = L*L^T
    multiplyLowerTriangular(L_opt, LLT_opt, n);
    double error_opt = matrixError(A, LLT_opt, n);
    
    double speedup = time_orig / time_opt;
    
    printf("Optimized algorithm:\n");
    printf("  Time taken: %f seconds\n", time_opt);
    printf("  Reconstruction error: %e\n", error_opt);
    printf("  Speedup: %.2f times\n", speedup);
    
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("L matrix (original):\n");
        printMatrix(L, n, n);
        printf("L matrix (optimized):\n");
        printMatrix(L_opt, n, n);
    }
    
    freeMatrix(L, n);
    freeMatrix(LLT, n);
    freeMatrix(L_opt, n);
    freeMatrix(LLT_opt, n);
    
    return speedup;
}

// Test LU decomposition with partial pivoting (original and optimized)
double testPartialPivotingLU(double **A, int n) {
    printf("\n--- Testing LU decomposition with Partial Pivoting ---\n");
    
    // For original algorithm
    double **L = allocateMatrix(n);
    double **U = allocateMatrix(n);
    double **PA = allocateMatrix(n);
    double **LU = allocateMatrix(n);
    int *P = (int *)malloc(n * sizeof(int));
    
    // Make a copy of A to avoid modifying it
    double **A_copy = allocateMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_copy[i][j] = A[i][j];
        }
    }
    
    // Measure time for original algorithm
    clock_t start = clock();
    PartialPivotingLU(A_copy, L, U, P, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Apply permutation to A to get PA
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            PA[i][j] = A[P[i]][j];
        }
    }
    
    // Verify result: PA = L*U
    multiplyMatrices(L, U, LU, n);
    double error_orig = matrixError(PA, LU, n);
    
    printf("Original algorithm:\n");
    printf("  Time taken: %f seconds\n", time_orig);
    printf("  Reconstruction error: %e\n", error_orig);
    
    // For optimized algorithm
    double **L_opt = allocateMatrix(n);
    double **U_opt = allocateMatrix(n);
    double **PA_opt = allocateMatrix(n);
    double **LU_opt = allocateMatrix(n);
    int *P_opt = (int *)malloc(n * sizeof(int));
    
    // Make another copy of A
    double **A_copy_opt = allocateMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_copy_opt[i][j] = A[i][j];
        }
    }
    
    // Measure time for optimized algorithm
    start = clock();
    PartialPivotingLU_optimized(A_copy_opt, L_opt, U_opt, P_opt, n);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Apply permutation to A to get PA
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            PA_opt[i][j] = A[P_opt[i]][j];
        }
    }
    
    // Verify result: PA = L*U
    multiplyMatrices(L_opt, U_opt, LU_opt, n);
    double error_opt = matrixError(PA_opt, LU_opt, n);
    
    double speedup = time_orig / time_opt;
    
    printf("Optimized algorithm:\n");
    printf("  Time taken: %f seconds\n", time_opt);
    printf("  Reconstruction error: %e\n", error_opt);
    printf("  Speedup: %.2f times\n", speedup);
    
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("Permutation P (original): ");
        for (int i = 0; i < n; i++) printf("%d ", P[i]);
        printf("\nPermutation P (optimized): ");
        for (int i = 0; i < n; i++) printf("%d ", P_opt[i]);
        printf("\n");
        printf("L matrix (original):\n");
        printMatrix(L, n, n);
        printf("U matrix (original):\n");
        printMatrix(U, n, n);
        printf("L matrix (optimized):\n");
        printMatrix(L_opt, n, n);
        printf("U matrix (optimized):\n");
        printMatrix(U_opt, n, n);
    }
    
    free(P);
    free(P_opt);
    freeMatrix(L, n);
    freeMatrix(U, n);
    freeMatrix(PA, n);
    freeMatrix(LU, n);
    freeMatrix(A_copy, n);
    freeMatrix(L_opt, n);
    freeMatrix(U_opt, n);
    freeMatrix(PA_opt, n);
    freeMatrix(LU_opt, n);
    freeMatrix(A_copy_opt, n);
    
    return speedup;
}

// Test LDL^T decomposition (original and optimized)
double testLDLTDecomposition(double **A, int n) {
    printf("\n--- Testing LDL^T decomposition ---\n");
    
    // For original algorithm
    double **L = allocateMatrix(n);
    double *D = (double *)malloc(n * sizeof(double));
    double **LDLT = allocateMatrix(n);
    
    // Measure time for original algorithm
    clock_t start = clock();
    int result = LDLTDecomposition(A, L, D, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result == -1) {
        printf("Original algorithm: LDL^T decomposition failed - Matrix is not symmetric or decomposition failed\n");
        freeMatrix(L, n);
        free(D);
        freeMatrix(LDLT, n);
        return -1.0;
    }
    
    // Verify result: A = L*D*L^T
    // First compute L*D
    double **LD = allocateMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LD[i][j] = L[i][j] * (j == i ? D[j] : (j < i ? 1.0 : 0.0));
        }
    }
    
    // Then compute (L*D)*L^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LDLT[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                if (k <= i && k <= j) { // Only where both L and L^T have non-zeros
                    LDLT[i][j] += LD[i][k] * L[j][k];
                }
            }
        }
    }
    
    double error_orig = matrixError(A, LDLT, n);
    
    printf("Original algorithm:\n");
    printf("  Time taken: %f seconds\n", time_orig);
    printf("  Reconstruction error: %e\n", error_orig);
    
    // For optimized algorithm
    double **L_opt = allocateMatrix(n);
    double *D_opt = (double *)malloc(n * sizeof(double));
    double **LDLT_opt = allocateMatrix(n);
    
    // Measure time for optimized algorithm
    start = clock();
    int result_opt = LDLTDecomposition_optimized(A, L_opt, D_opt, n);
    end = clock();
    double time_opt = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result_opt == -1) {
        printf("Optimized algorithm: LDL^T decomposition failed - Matrix is not symmetric or decomposition failed\n");
        freeMatrix(L, n);
        free(D);
        freeMatrix(LDLT, n);
        freeMatrix(LD, n);
        freeMatrix(L_opt, n);
        free(D_opt);
        freeMatrix(LDLT_opt, n);
        return -1.0;
    }
    
    // Verify result: A = L*D*L^T for optimized version
    double **LD_opt = allocateMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LD_opt[i][j] = L_opt[i][j] * (j == i ? D_opt[j] : (j < i ? 1.0 : 0.0));
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LDLT_opt[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                if (k <= i && k <= j) {
                    LDLT_opt[i][j] += LD_opt[i][k] * L_opt[j][k];
                }
            }
        }
    }
    
    double error_opt = matrixError(A, LDLT_opt, n);
    
    double speedup = time_orig / time_opt;
    
    printf("Optimized algorithm:\n");
    printf("  Time taken: %f seconds\n", time_opt);
    printf("  Reconstruction error: %e\n", error_opt);
    printf("  Speedup: %.2f times\n", speedup);
    
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("L matrix (original):\n");
        printMatrix(L, n, n);
        printf("D diagonal (original): ");
        for (int i = 0; i < n; i++) printf("%8.4f ", D[i]);
        printf("\nL matrix (optimized):\n");
        printMatrix(L_opt, n, n);
        printf("D diagonal (optimized): ");
        for (int i = 0; i < n; i++) printf("%8.4f ", D_opt[i]);
        printf("\n");
    }
    
    freeMatrix(L, n);
    free(D);
    freeMatrix(LDLT, n);
    freeMatrix(LD, n);
    freeMatrix(L_opt, n);
    free(D_opt);
    freeMatrix(LDLT_opt, n);
    freeMatrix(LD_opt, n);
    
    return speedup;
}