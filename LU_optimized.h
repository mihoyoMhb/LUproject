#ifndef LU_OPTIMIZED_H
#define LU_OPTIMIZED_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Optimized standard LU decomposition of a matrix A into L and U
 * Uses cache-friendly access patterns and reduces division operations
 * With vectorization hints for compiler
 * @param A Input matrix (n x n)
 * @param L Lower triangular matrix with diagonal elements = 1
 * @param U Upper triangular matrix
 * @param n Matrix dimension
 */
void LUdecomposition_optimized(double * restrict const * restrict A, 
                              double * restrict * restrict L, 
                              double * restrict * restrict U, 
                              const int n);

/**
 * Optimized Cholesky decomposition for symmetric positive definite matrices
 * Uses cache-friendly access patterns and minimizes square root operations
 * With vectorization hints for compiler
 * @param A Input matrix (n x n), must be symmetric positive definite
 * @param L Lower triangular matrix
 * @param n Matrix dimension
 * @return 0 if successful, -1 if matrix is not positive definite
 */
int CholeskyDecomposition_optimized(double * restrict const * restrict A, 
                                   double * restrict * restrict L, 
                                   const int n);

/**
 * Optimized LU decomposition with partial pivoting
 * Uses blocking and minimizes memory access patterns
 * With vectorization hints for compiler
 * @param A Input matrix (n x n)
 * @param L Lower triangular matrix
 * @param U Upper triangular matrix
 * @param P Array representing permutation matrix (row exchanges)
 * @param n Matrix dimension
 */
void PartialPivotingLU_optimized(double * restrict const * restrict A, 
                                double * restrict * restrict L, 
                                double * restrict * restrict U, 
                                int * restrict P, 
                                const int n);


#endif /* LU_OPTIMIZED_H */