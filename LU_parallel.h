#ifndef LU_PARALLEL_H
#define LU_PARALLEL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/**
 * Parallel LU decomposition of a matrix A into L and U using OpenMP
 * @param A Input matrix (n x n)
 * @param L Lower triangular matrix with diagonal elements = 1
 * @param U Upper triangular matrix
 * @param n Matrix dimension
 * @param num_threads Number of threads to use
 */
void LUdecomposition_parallel(double **A, double **L, double **U, int n, int num_threads);

/**
 * Parallel Cholesky decomposition for symmetric positive definite matrices (A = L*L^T) using OpenMP
 * @param A Input matrix (n x n), must be symmetric positive definite
 * @param L Lower triangular matrix
 * @param n Matrix dimension
 * @param num_threads Number of threads to use
 * @return 0 if successful, -1 if matrix is not positive definite
 */
int CholeskyDecomposition_parallel(double **A, double **L, int n, int num_threads);

/**
 * Parallel LU decomposition with partial pivoting using OpenMP
 * @param A Input matrix (n x n)
 * @param L Lower triangular matrix
 * @param U Upper triangular matrix
 * @param P Array representing permutation matrix (row exchanges)
 * @param n Matrix dimension
 * @param num_threads Number of threads to use
 */
void PartialPivotingLU_parallel(double **A, double **L, double **U, int *P, int n, int num_threads);

/**
 * Parallel LDL^T decomposition for symmetric matrices using OpenMP
 * @param A Input matrix (n x n), must be symmetric
 * @param L Lower triangular matrix with diagonal elements = 1
 * @param D Array representing diagonal matrix
 * @param n Matrix dimension
 * @param num_threads Number of threads to use
 * @return 0 if successful, -1 if matrix is not symmetric or decomposition failed
 */
int LDLTDecomposition_parallel(double **A, double **L, double *D, int n, int num_threads);

#endif /* LU_PARALLEL_H */