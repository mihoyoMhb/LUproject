#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

//gcc -o Parallel Parallel.c -O3 -march=native -fopenmp -ffast-math -ftree-vectorize -fopt-info-vec
/* Parallel optimized LU decomposition: Using inner parallelization, outer loop sequential */
void LUdecomposition_optimized_parallel(double * restrict const * restrict A, 
                                          double * restrict * restrict L, 
                                          double * restrict * restrict U, 
                                          const int n) {
    // Parallel initialization: L as identity matrix, U as zero matrix
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }
    
    // LU decomposition: outer loop processes each column sequentially
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for schedule(static) default(none) shared(A, L, U, n, i)
        for (int j = i; j < n; j++) {
            double sum = A[i][j];
            for (int k = 0; k < i; k++) {
                sum -= L[i][k] * U[k][j];
            }
            U[i][j] = sum;
        }
        
        // Precompute reciprocal of U[i][i]
        double u_ii_inv = 1.0 / U[i][i];
        
        // Calculate L's i-th column (from i+1 to n-1) rows, can be parallelized
        #pragma omp parallel for schedule(static) default(none) shared(A, L, U, n, i, u_ii_inv)
        for (int j = i + 1; j < n; j++) {
            double sum = A[j][i];
            for (int k = 0; k < i; k++) {
                sum -= L[j][k] * U[k][i];
            }
            L[j][i] = sum * u_ii_inv;
        }
    }
}

/* Parallel optimized Cholesky decomposition: requires A to be symmetric positive definite, result satisfies A = L * L^T */
int CholeskyDecomposition_optimized_parallel(double * restrict const * restrict A, 
                                               double * restrict * restrict L, 
                                               const int n) {
   
    /*
    https://en.wikipedia.org/wiki/Cholesky_decomposition
    https://www.geeksforgeeks.org/cholesky-decomposition-matrix-decomposition/
    */    
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            L[i][j] = 0.0;
        }
    }
    
    // Cholesky decomposition: outer loop processes each column sequentially
    for (int j = 0; j < n; j++) {
        double sum_diag = A[j][j];
        // Parallel computation of diagonal update accumulation (using reduction)
        #pragma omp parallel for reduction(-:sum_diag) schedule(static) default(none) shared(L, j)
        for (int k = 0; k < j; k++) {
            double L_jk = L[j][k];
            sum_diag -= L_jk * L_jk;
        }
        
        if (sum_diag <= 0.0) {
            return -1;  // Matrix is not positive definite
        }
        
        L[j][j] = sqrt(sum_diag);
        double inv_L_jj = 1.0 / L[j][j];
        
        // Calculate remaining elements of j-th column (i = j+1..n-1) in parallel
        #pragma omp parallel for schedule(static) default(none) shared(A, L, n, j, inv_L_jj)
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

/* ===== Helper Functions ===== */

// Allocate n x n matrix (double type)
double **allocate_matrix(int n) {
    double **mat = (double **) aligned_alloc(64, n * sizeof(double *));
    // Allocate one contiguous block of memory
    double *data = (double *) aligned_alloc(64, n * n * sizeof(double));
    
    // Set up pointers to the rows
    for (int i = 0; i < n; i++) {
        mat[i] = &data[i * n];
    }
    return mat;
}

// Free n x n matrix
void free_matrix(double **mat, int n) {
    free(mat[0]);  // Free the data block
    free(mat);     // Free the row pointers
}

// Initialize matrix with random numbers ([0,1])
void randomMatrix(double **A, int n) {
    srand(42); // Seed the random number generator
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)rand() / RAND_MAX * 20.0 - 10.0; // Values between -10 and 10
        }
    }
}


// Generate symmetric positive definite matrix A = B * B^T, where B is a random matrix
void generate_spd_matrix(double **A, int n) {
    double **B = allocate_matrix(n);
    
    // Initialize B with random values in parallel
    randomMatrix(B, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            // Use vectorizable loop
            for (int k = 0; k < n; k++) {
                sum += B[i][k] * B[j][k];
            }
            A[i][j] = sum;
        }
    }

    for (int i = 0; i < n; i++) {
        A[i][i] += 1.0;
    }
    
    free_matrix(B, n);
}

// Matrix multiplication: compute C = A * B
void matrix_multiply(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++){
        for(int k = 0; k < n; k++){
            //ikj
            double a_ik = A[i][k];
            for (int j = 0; j < n; j++){
                C[i][j] += a_ik * B[k][j];
            }
        }
    }
}

// Calculate the Frobenius norm of the difference between two n x n matrices
double frobenius_norm_diff(double **A, double **B, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }
    return sqrt(sum);
}

/* ===== Test Main Function ===== */
int main(int argc, char *argv[]) {
    if(argc != 3){
        printf("Usage: %s <num_threads> <matrix_size>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    int matrix_size = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    srand(0);
    const int n = matrix_size;  // Matrix size

    /* ---- Test Parallel LU Decomposition ---- */
    double **A_lu = allocate_matrix(n);
    double **L_lu = allocate_matrix(n);
    double **U_lu = allocate_matrix(n);
    randomMatrix(A_lu, n);
    
    // Backup A_lu for error calculation
    double **A_lu_copy = allocate_matrix(n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A_lu_copy[i][j] = A_lu[i][j];
        }
    }
    
    double start = omp_get_wtime();
    LUdecomposition_optimized_parallel(A_lu, L_lu, U_lu, n);
    double end = omp_get_wtime();
    double lu_time = end - start;
    
    // Calculate L*U
    double **LU_product = allocate_matrix(n);
    matrix_multiply(L_lu, U_lu, LU_product, n);
    double lu_error = frobenius_norm_diff(A_lu_copy, LU_product, n);
    
    printf("Parallel LU Decomposition:\n");
    printf("Time: %f seconds\n", lu_time);
    printf("Frobenius norm error: %e\n", lu_error);
    
    free_matrix(A_lu, n);
    free_matrix(L_lu, n);
    free_matrix(U_lu, n);
    free_matrix(A_lu_copy, n);
    free_matrix(LU_product, n);
    
    /* ---- Test Parallel Cholesky Decomposition ---- */
    double **A_chol = allocate_matrix(n);
    double **L_chol = allocate_matrix(n);
    generate_spd_matrix(A_chol, n);
    
    // Backup A_chol for error calculation
    double **A_chol_copy = allocate_matrix(n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A_chol_copy[i][j] = A_chol[i][j];
        }
    }
    
    start = omp_get_wtime();
    int chol_status = CholeskyDecomposition_optimized_parallel(A_chol, L_chol, n);
    end = omp_get_wtime();
    double chol_time = end - start;
    
    if (chol_status != 0) {
        printf("\nParallel Cholesky decomposition failed: Matrix is not positive definite.\n");
    } else {
        // Calculate L * L^T
        double **LLT = allocate_matrix(n);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                double sum = 0.0;
                for (int k = 0; k < n; k++){
                    sum += L_chol[i][k] * L_chol[j][k];
                }
                LLT[i][j] = sum;
            }
        }
        double chol_error = frobenius_norm_diff(A_chol_copy, LLT, n);
        printf("\nParallel Cholesky Decomposition:\n");
        printf("Time: %f seconds\n", chol_time);
        printf("Frobenius norm error: %e\n", chol_error);
        free_matrix(LLT, n);
    }
    
    free_matrix(A_chol, n);
    free_matrix(L_chol, n);
    free_matrix(A_chol_copy, n);
    
    return 0;
}