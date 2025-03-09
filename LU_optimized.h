#ifndef LU_OPTIMIZED_H
#define LU_OPTIMIZED_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Optimized standard LU decomposition of a matrix A into L and U
 * Uses cache-friendly access patterns and reduces division operations
 * @param A Input matrix (n x n)
 * @param L Lower triangular matrix with diagonal elements = 1
 * @param U Upper triangular matrix
 * @param n Matrix dimension
 */
void LUdecomposition_optimized(double **A, double **L, double **U, int n);

/**
 * Optimized Cholesky decomposition for symmetric positive definite matrices
 * Uses cache-friendly access patterns and minimizes square root operations
 * @param A Input matrix (n x n), must be symmetric positive definite
 * @param L Lower triangular matrix
 * @param n Matrix dimension
 * @return 0 if successful, -1 if matrix is not positive definite
 */
int CholeskyDecomposition_optimized(double **A, double **L, int n);

/**
 * Optimized LU decomposition with partial pivoting
 * Uses blocking and minimizes memory access patterns
 * @param A Input matrix (n x n)
 * @param L Lower triangular matrix
 * @param U Upper triangular matrix
 * @param P Array representing permutation matrix (row exchanges)
 * @param n Matrix dimension
 */
void PartialPivotingLU_optimized(double **A, double **L, double **U, int *P, int n);

/**
 * Optimized LDL^T decomposition for symmetric matrices
 * Uses symmetric properties to reduce computation
 * @param A Input matrix (n x n), must be symmetric
 * @param L Lower triangular matrix with diagonal elements = 1
 * @param D Array representing diagonal matrix
 * @param n Matrix dimension
 * @return 0 if successful, -1 if matrix is not symmetric or decomposition failed
 */
int LDLTDecomposition_optimized(double **A, double **L, double *D, int n);

#endif /* LU_OPTIMIZED_H */