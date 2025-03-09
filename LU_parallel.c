#include "LU_parallel.h"

/* Parallel LU decomposition using OpenMP */
void LUdecomposition_parallel(double **A, double **L, double **U, int n, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Initialize L as identity matrix and U as zero matrix
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }
    
    // Main decomposition loop - can't parallelize outer loop due to dependencies
    for (int i = 0; i < n; i++) {
        // Calculate U's i-th row - can be parallelized
        #pragma omp parallel for schedule(dynamic)
        for (int j = i; j < n; j++) {
            double sum = A[i][j];
            for (int k = 0; k < i; k++) {
                sum -= L[i][k] * U[k][j];
            }
            U[i][j] = sum;
        }
        
        // Calculate L's i-th column - can be parallelized
        // Pre-compute the inverse of U[i][i] outside the parallel region to avoid race conditions
        double u_ii_inv = 1.0 / U[i][i];
        
        #pragma omp parallel for schedule(dynamic)
        for (int j = i + 1; j < n; j++) {
            double sum = A[j][i];
            // This inner loop has a shorter range as i increases, good for dynamic scheduling
            for (int k = 0; k < i; k++) {
                sum -= L[j][k] * U[k][i];
            }
            L[j][i] = sum * u_ii_inv;
        }
    }
}

/* Parallel Cholesky decomposition using OpenMP */
int CholeskyDecomposition_parallel(double **A, double **L, int n, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Initialize L to 0
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = 0.0;
        }
    }
    
    // Cholesky decomposition - outer loop has dependencies
    for (int j = 0; j < n; j++) {
        double sum_diag = A[j][j];
        
        // This reduction can be parallelized
        #pragma omp parallel for reduction(-:sum_diag) schedule(dynamic, 16)
        for (int k = 0; k < j; k++) {
            double L_jk = L[j][k];
            sum_diag -= L_jk * L_jk;
        }
        
        if (sum_diag <= 0.0) {
            return -1;  // Not positive definite
        }
        
        L[j][j] = sqrt(sum_diag);
        double inv_L_jj = 1.0 / L[j][j];
        
        // Compute remaining elements in column j - can be parallelized
        #pragma omp parallel for schedule(dynamic)
        for (int i = j + 1; i < n; i++) {
            double sum = A[i][j];
            // Manual loop unrolling for better vectorization
            for (int k = 0; k < j - 3; k += 4) {
                sum -= L[i][k] * L[j][k];
                sum -= L[i][k+1] * L[j][k+1];
                sum -= L[i][k+2] * L[j][k+2];
                sum -= L[i][k+3] * L[j][k+3];
            }
            // Handle remaining elements
            for (int k = j - (j % 4); k < j; k++) {
                sum -= L[i][k] * L[j][k];
            }
            L[i][j] = sum * inv_L_jj;
        }
    }
    return 0;
}

/* Parallel LU decomposition with partial pivoting using OpenMP */
void PartialPivotingLU_parallel(double **A, double **L, double **U, int *P, int n, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Initialize P as [0,1,...,n-1]
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }
    
    // Make a copy of A to avoid modifying the original
    double **tempA = (double **)malloc(n * sizeof(double *));
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        tempA[i] = (double *)malloc(n * sizeof(double));
        // Initialize L and U, copy A to tempA
        for (int j = 0; j < n; j++) {
            tempA[i][j] = A[i][j];
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }
    
    // Process the matrix column by column - outer loop must be sequential
    for (int k = 0; k < n; k++) {
        // Find pivot - can be parallelized with reduction
        int max_row = k;
        double max_val = fabs(tempA[k][k]);
        
        #pragma omp parallel
        {
            int local_max_row = k;
            double local_max_val = max_val;
            
            #pragma omp for schedule(dynamic)
            for (int i = k + 1; i < n; i++) {
                double abs_val = fabs(tempA[i][k]);
                if (abs_val > local_max_val) {
                    local_max_val = abs_val;
                    local_max_row = i;
                }
            }
            
            // Combine local results
            #pragma omp critical
            {
                if (local_max_val > max_val) {
                    max_val = local_max_val;
                    max_row = local_max_row;
                }
            }
        }
        
        // Swap rows if needed - must be done sequentially
        if (max_row != k) {
            // Swap tempA rows efficiently
            double *tmp_row = tempA[k];
            tempA[k] = tempA[max_row];
            tempA[max_row] = tmp_row;
            
            // Swap P entries
            int temp_p = P[k];
            P[k] = P[max_row];
            P[max_row] = temp_p;
            
            // Swap already calculated parts of L (only columns 0 to k-1)
            #pragma omp parallel for schedule(static) if(k > 16)
            for (int j = 0; j < k; j++) {
                double temp_l = L[k][j];
                L[k][j] = L[max_row][j];
                L[max_row][j] = temp_l;
            }
        }
        
        // Calculate U's k-th row - can be parallelized
        #pragma omp parallel for schedule(dynamic)
        for (int j = k; j < n; j++) {
            double sum = tempA[k][j];
            for (int m = 0; m < k; m++) {
                sum -= L[k][m] * U[m][j];
            }
            U[k][j] = sum;
        }
        
        // Calculate L's k-th column - can be parallelized
        // Precompute division factor
        double u_kk_inv = 0.0;
        if (U[k][k] != 0.0) {
            u_kk_inv = 1.0 / U[k][k];
            
            #pragma omp parallel for schedule(dynamic)
            for (int i = k + 1; i < n; i++) {
                double sum = tempA[i][k];
                // Use simd directive for vectorization of inner loop
                #pragma omp simd reduction(-:sum)
                for (int m = 0; m < k; m++) {
                    sum -= L[i][m] * U[m][k];
                }
                L[i][k] = sum * u_kk_inv;
            }
        }
    }
    
    // Free temporary matrix
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        free(tempA[i]);
    }
    free(tempA);
}

/* Parallel LDL^T decomposition using OpenMP */
int LDLTDecomposition_parallel(double **A, double **L, double *D, int n, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // First check if matrix is symmetric - can be parallelized
    int is_symmetric = 1;
    #pragma omp parallel for schedule(dynamic) reduction(&:is_symmetric)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (A[i][j] != A[j][i]) {
                is_symmetric = 0;
            }
        }
    }
    
    if (!is_symmetric) return -1;
    
    // Initialize L as identity matrix
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Process the matrix column by column - outer loop must be sequential
    for (int j = 0; j < n; j++) {
        // Calculate D[j] - can use parallel reduction
        double d_sum = A[j][j];
        
        #pragma omp parallel for reduction(-:d_sum) schedule(dynamic, 16)
        for (int k = 0; k < j; k++) {
            double L_jk = L[j][k];
            d_sum -= L_jk * L_jk * D[k];
        }
        
        if (d_sum == 0.0) return -1;  // Matrix is singular
        D[j] = d_sum;
        
        // Calculate column j of L - can be parallelized
        double d_j_inv = 1.0 / D[j];
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = j + 1; i < n; i++) {
            double l_sum = A[i][j];
            // Parallelize inner reduction with simd
            #pragma omp simd reduction(-:l_sum)
            for (int k = 0; k < j; k++) {
                l_sum -= L[i][k] * L[j][k] * D[k];
            }
            L[i][j] = l_sum * d_j_inv;
        }
    }
    
    return 0;
}