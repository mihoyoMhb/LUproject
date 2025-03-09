#ifndef LU_DECOMPOSITION_H
#define LU_DECOMPOSITION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * Standard LU decomposition of a matrix A into L and U
 * @param A Input matrix (n x n)
 * @param L Lower triangular matrix with diagonal elements = 1
 * @param U Upper triangular matrix
 * @param n Matrix dimension
 */
void LUdecomposition(double **A, double **L, double **U, int n);

/**
 * Cholesky decomposition for symmetric positive definite matrices (A = L*L^T)
 * @param A Input matrix (n x n), must be symmetric positive definite
 * @param L Lower triangular matrix
 * @param n Matrix dimension
 * @return 0 if successful, -1 if matrix is not positive definite
 */
int CholeskyDecomposition(double **A, double **L, int n);

/**
 * LU decomposition with partial pivoting for better numerical stability
 * @param A Input matrix (n x n)
 * @param L Lower triangular matrix
 * @param U Upper triangular matrix
 * @param P Array representing permutation matrix (row exchanges)
 * @param n Matrix dimension
 */
void PartialPivotingLU(double **A, double **L, double **U, int *P, int n);

/**
 * LDL^T decomposition for symmetric matrices
 * @param A Input matrix (n x n), must be symmetric
 * @param L Lower triangular matrix with diagonal elements = 1
 * @param D Array representing diagonal matrix
 * @param n Matrix dimension
 * @return 0 if successful, -1 if matrix is not symmetric or decomposition failed
 */
int LDLTDecomposition(double **A, double **L, double *D, int n);

/**
 * Print a matrix to console
 * @param M Matrix to print
 * @param n Number of rows
 * @param m Number of columns
 */
void printMatrix(double **M, int n, int m);

/**
 * Compute A = L*L^T for a lower triangular matrix L
 * @param L_true Lower triangular matrix
 * @param A Result matrix
 * @param n Matrix dimension
 */
void multiplyLowerTriangular(double **L_true, double **A, int n);

#endif /* LU_DECOMPOSITION_H */