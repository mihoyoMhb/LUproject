/*File name: situ_parallel_naive.c*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
//gcc -o situ situ_serial.c -O3 -march=native -fopenmp -ffast-math -ftree-vectorize
//gcc -o situ situ_serial.c -O2 -fopenmp -lm
// Helper function to allocate a matrix
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
// Free matrix
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

double verify_lu(double **A_orig, double **LU, int n) {
    // 分配 L, U 以及重构矩阵 P
    double **L = allocate_matrix(n);
    double **U = allocate_matrix(n);
    double **P = allocate_matrix(n);

    // 初始化 L 和 U：从 LU 中提取下三角和上三角
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (i > j) {            // 下三角：L 的值存储在 LU 中
                L[i][j] = LU[i][j];
                U[i][j] = 0.0;
            } else if (i == j) {      // 对角线：L 对角为 1，U 对角从 LU 中取
                L[i][j] = 1.0;
                U[i][j] = LU[i][j];
            } else {                  // i < j，上三角：U 的值存储在 LU 中
                L[i][j] = 0.0;
                U[i][j] = LU[i][j];
            }
            // 初始化 P（用于存储 L*U 的乘积）为 0
            P[i][j] = 0.0;
        }
    }

    // 计算矩阵乘法 P = L * U
    for (int i = 0; i < n; i++){
        for (int k = 0; k < n; k++){
            double l_ik = L[i][k];
            for (int j = 0; j < n; j++){
                P[i][j] += l_ik * U[k][j];
            }
        }
    }

    // 计算 Frobenius 范数误差： sqrt(sum((A_orig - P)^2))
    double error = 0.0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double d = A_orig[i][j] - P[i][j];
            error += d * d;
        }
    }
    error = sqrt(error);

    free_matrix(L, n);
    free_matrix(U, n);
    free_matrix(P, n);

    return error;
}

void lu_in_situ_ver0(double ** restrict A, int n) {
    for (int k = 0; k < n; k++) {
        double temp = 1/A[k][k];
        for (int i = k + 1; i < n; i++) {
            A[i][k] *= temp;
        }
    #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}




int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <num_threads> <matrix_size>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    int matrix_size = atoi(argv[2]);
    omp_set_num_threads(num_threads);
    srand(0);
    int n = matrix_size;  // Matrix size
    double start_time, end_time, lu_time;


    double **A = allocate_matrix(n);
    randomMatrix(A, n);


    start_time = omp_get_wtime();
    lu_in_situ_ver0(A, n);
    end_time = omp_get_wtime();
    lu_time = end_time - start_time;
    printf("In-situ LU factorization time (Naive parallel): %f seconds\n", lu_time);


    free_matrix(A, n);

    return 0;
}