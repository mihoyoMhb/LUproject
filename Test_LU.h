#ifndef TEST_LU_H
#define TEST_LU_H

#include "LU_decomposition.h"
#include "LU_optimized.h"

/**
 * Allocate a dynamic 2D matrix
 * @param n Matrix dimension
 * @return Allocated matrix
 */
double** allocateMatrix(int n);

/**
 * Free a dynamic 2D matrix
 * @param matrix Matrix to free
 * @param n Matrix dimension
 */
void freeMatrix(double **matrix, int n);

/**
 * Create a random matrix
 * @param A Output matrix
 * @param n Matrix dimension
 */
void randomMatrix(double **A, int n);

/**
 * Create a symmetric matrix
 * @param A Output matrix
 * @param n Matrix dimension
 */
void symmetricMatrix(double **A, int n);

/**
 * Create a symmetric positive definite matrix
 * @param A Output matrix
 * @param n Matrix dimension
 */
void positiveDefiniteMatrix(double **A, int n);

/**
 * Multiply two matrices C = A*B
 * @param A First matrix
 * @param B Second matrix
 * @param C Result matrix
 * @param n Matrix dimension
 */
void multiplyMatrices(double **A, double **B, double **C, int n);

/**
 * Calculate Frobenius norm of difference between two matrices
 * @param A First matrix
 * @param B Second matrix
 * @param n Matrix dimension
 * @return Error value
 */
double matrixError(double **A, double **B, int n);

/**
 * Test standard LU decomposition (original and optimized)
 * @param A Input matrix
 * @param n Matrix dimension
 * @return Speedup ratio (optimized/original) or -1 if test failed
 */
double testLUdecomposition(double **A, int n);

/**
 * Test Cholesky decomposition (original and optimized)
 * @param A Input matrix
 * @param n Matrix dimension
 * @return Speedup ratio (optimized/original) or -1 if test failed
 */
double testCholeskyDecomposition(double **A, int n);

/**
 * Test LU decomposition with partial pivoting (original and optimized)
 * @param A Input matrix
 * @param n Matrix dimension
 * @return Speedup ratio (optimized/original) or -1 if test failed
 */
double testPartialPivotingLU(double **A, int n);

/**
 * Test LDL^T decomposition (original and optimized)
 * @param A Input matrix
 * @param n Matrix dimension
 * @return Speedup ratio (optimized/original) or -1 if test failed
 */
double testLDLTDecomposition(double **A, int n);

#endif /* TEST_LU_H */