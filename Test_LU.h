#ifndef TEST_LU_H
#define TEST_LU_H

#include "LU_decomposition.h"
#include "LU_optimized.h"

// Matrix handling functions
double** allocateMatrix(int n);
void freeMatrix(double **matrix, int n);
void randomMatrix(double **A, int n);
void symmetricMatrix(double **A, int n);
void positiveDefiniteMatrix(double **A, int n);
void multiplyMatrices(double **A, double **B, double **C, int n);
double matrixError(double **A, double **B, int n);


/**
 * Test different implementations of LU decomposition
 * @param A Input matrix
 * @param n Matrix dimension
 */
void testAllLU(double **A, int n);

/**
 * Test different implementations of Cholesky decomposition
 * @param A Input matrix
 * @param n Matrix dimension
 */
void testAllCholesky(double **A, int n);

/**
 * Test different implementations of LU with partial pivoting
 * @param A Input matrix
 * @param n Matrix dimension
 */
void testAllPivotLU(double **A, int n);


#endif /* TEST_LU_H */