#include "LU_optimized.h"

/* Optimized LU decomposition */
void LUdecomposition_optimized(double * restrict const * restrict A, 
                              double * restrict * restrict L, 
                              double * restrict * restrict U, 
                              const int n) {
    // Initialize L as identity matrix and U as zero matrix
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }
    
    // Cache-friendly implementation with loop reordering
    for (int i = 0; i < n; i++) {
        // Pre-compute U's i-th row
        #pragma omp simd
        for (int j = i; j < n; j++) {
            double sum = A[i][j];
            for (int k = 0; k < i; k++) {
                sum -= L[i][k] * U[k][j];
            }
            U[i][j] = sum;
        }
        
        // Use the computed U values to calculate L's i-th column
        double u_ii_inv = 1.0 / U[i][i]; // Precompute division
        
        #pragma omp simd
        for (int j = i + 1; j < n; j++) {
            double sum = A[j][i];
            for (int k = 0; k < i; k++) {
                sum -= L[j][k] * U[k][i];
            }
            L[j][i] = sum * u_ii_inv; // Multiplication instead of division
        }
    }
}

/* Optimized Cholesky decomposition */
int CholeskyDecomposition_optimized(double * restrict const * restrict A, 
                                  double * restrict * restrict L, 
                                  const int n) {
    // Initialize L to 0
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = 0.0;
        }
    }
    
    // Optimized computation with reduced square roots and better cache access
    for (int j = 0; j < n; j++) {
        double sum_diag = A[j][j];
        
        // Update diagonal element j,j
        #pragma omp simd reduction(-:sum_diag)
        for (int k = 0; k < j; k++) {
            double L_jk = L[j][k];
            sum_diag -= L_jk * L_jk;
        }
        
        if (sum_diag <= 0.0) {
            return -1;  // Not positive definite
        }
        
        L[j][j] = sqrt(sum_diag);
        double inv_L_jj = 1.0 / L[j][j];
        
        // Compute remaining elements in column j
        #pragma omp simd
        for (int i = j + 1; i < n; i++) {
            double sum = A[i][j];
            for (int k = 0; k < j; k++) {
                sum -= L[i][k] * L[j][k];
            }
            L[i][j] = sum * inv_L_jj;
        }
    }
    return 0;
}

/* Optimized LU decomposition with partial pivoting */
void PartialPivotingLU_optimized(double * restrict const * restrict A, 
                               double * restrict * restrict L, 
                               double * restrict * restrict U, 
                               int * restrict P, 
                               const int n) {
    // Initialize P as [0,1,...,n-1]
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }
    
    // Make a copy of A to avoid modifying the original
    double ** restrict tempA = (double **)malloc(n * sizeof(double *));
    
    for (int i = 0; i < n; i++) {
        tempA[i] = (double *)malloc(n * sizeof(double));
        // Initialize L and U, copy A to tempA
        #pragma omp simd
        for (int j = 0; j < n; j++) {
            tempA[i][j] = A[i][j];
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }
    
    // Process the matrix column by column
    for (int k = 0; k < n; k++) {
        // Find pivot (max element in column k from row k to n-1)
        int max_row = k;
        double max_val = fabs(tempA[k][k]);
        
        for (int i = k + 1; i < n; i++) {
            double abs_val = fabs(tempA[i][k]);
            if (abs_val > max_val) {
                max_val = abs_val;
                max_row = i;
            }
        }
        
        // Swap rows if needed
        if (max_row != k) {
            // Swap tempA rows efficiently
            double * restrict tmp_row = tempA[k];
            tempA[k] = tempA[max_row];
            tempA[max_row] = tmp_row;
            
            // Swap P entries
            int temp_p = P[k];
            P[k] = P[max_row];
            P[max_row] = temp_p;
            
            // Swap already calculated parts of L (only columns 0 to k-1)
            #pragma omp simd
            for (int j = 0; j < k; j++) {
                double temp_l = L[k][j];
                L[k][j] = L[max_row][j];
                L[max_row][j] = temp_l;
            }
        }
        
        // Calculate U's k-th row
        #pragma omp simd
        for (int j = k; j < n; j++) {
            double sum = tempA[k][j];
            for (int m = 0; m < k; m++) {
                sum -= L[k][m] * U[m][j];
            }
            U[k][j] = sum;
        }
        
        // Calculate L's k-th column - precompute division
        if (U[k][k] != 0.0) {  // Avoid division by zero
            double u_kk_inv = 1.0 / U[k][k];
            
            #pragma omp simd
            for (int i = k + 1; i < n; i++) {
                double sum = tempA[i][k];
                for (int m = 0; m < k; m++) {
                    sum -= L[i][m] * U[m][k];
                }
                L[i][k] = sum * u_kk_inv;
            }
        }
    }
    
    // Free temporary matrix
    for (int i = 0; i < n; i++) {
        free(tempA[i]);
    }
    free(tempA);
}

/* Optimized LDL^T decomposition */
int LDLTDecomposition_optimized(double * restrict const * restrict A, 
                              double * restrict * restrict L, 
                              double * restrict D, 
                              const int n) {
    // First check if matrix is symmetric (only check upper triangle)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (A[i][j] != A[j][i]) return -1;
        }
    }
    
    // Initialize L as identity matrix
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Process the matrix column by column with improved cache usage
    for (int j = 0; j < n; j++) {
        // Calculate D[j]
        double d_sum = A[j][j];
        
        #pragma omp simd reduction(-:d_sum)
        for (int k = 0; k < j; k++) {
            double L_jk_squared = L[j][k] * L[j][k];
            d_sum -= L_jk_squared * D[k];
        }
        
        if (d_sum == 0.0) return -1;  // Matrix is singular
        D[j] = d_sum;
        
        // Calculate column j of L
        double d_j_inv = 1.0 / D[j];
        
        #pragma omp simd
        for (int i = j + 1; i < n; i++) {
            double l_sum = A[i][j];
            for (int k = 0; k < j; k++) {
                l_sum -= L[i][k] * L[j][k] * D[k];
            }
            L[i][j] = l_sum * d_j_inv;
        }
    }
    
    return 0;
}