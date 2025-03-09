#ifndef TEST_LU_H
#define TEST_LU_H

#include "LU_decomposition.h"
#include "LU_optimized.h"
#include "LU_parallel.h"

// Matrix handling functions
double** allocateMatrix(int n);
void freeMatrix(double **matrix, int n);
void randomMatrix(double **A, int n);
void symmetricMatrix(double **A, int n);
void positiveDefiniteMatrix(double **A, int n);
void multiplyMatrices(double **A, double **B, double **C, int n);
double matrixError(double **A, double **B, int n);

// Function pointer types for different decomposition methods
typedef void (*LUFunction)(double**, double**, double**, int);
typedef int (*CholeskyFunction)(double**, double**, int);
typedef void (*PivotLUFunction)(double**, double**, double**, int*, int);
typedef int (*LDLTFunction)(double**, double**, double*, int);

// Parallel function pointer types
typedef void (*LUFunctionP)(double**, double**, double**, int, int);
typedef int (*CholeskyFunctionP)(double**, double**, int, int);
typedef void (*PivotLUFunctionP)(double**, double**, double**, int*, int, int);
typedef int (*LDLTFunctionP)(double**, double**, double*, int, int);

/**
 * Test different implementations of LU decomposition
 * @param A Input matrix
 * @param n Matrix dimension
 * @param num_threads Number of threads for parallel implementation
 */
void testAllLU(double **A, int n, int num_threads);

/**
 * Test different implementations of Cholesky decomposition
 * @param A Input matrix
 * @param n Matrix dimension
 * @param num_threads Number of threads for parallel implementation
 */
void testAllCholesky(double **A, int n, int num_threads);

/**
 * Test different implementations of LU with partial pivoting
 * @param A Input matrix
 * @param n Matrix dimension
 * @param num_threads Number of threads for parallel implementation
 */
void testAllPivotLU(double **A, int n, int num_threads);

/**
 * Test different implementations of LDLT decomposition
 * @param A Input matrix
 * @param n Matrix dimension
 * @param num_threads Number of threads for parallel implementation
 */
void testAllLDLT(double **A, int n, int num_threads);

#endif /* TEST_LU_H */