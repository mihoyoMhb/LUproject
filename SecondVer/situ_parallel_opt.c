/*File name: situ_parallel_opt.c*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
//gcc -o situ situ_serial.c -O3 -march=native -fopenmp -ffast-math -ftree-vectorize
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

void lu_in_situ_serial(double ** restrict A, int n) {
    for (int k = 0; k < n; k++) {
        double temp = 1/A[k][k];
        for (int i = k + 1; i < n; i++) {
            A[i][k] *= temp;
        }
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}


// 使用 first touch 策略并行初始化矩阵
void parallel_initialize_matrix(double **A, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        // 为每一行设置线程私有的随机种子，确保随机数生成独立且能 first touch
        unsigned int seed = 42 + i;
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)rand_r(&seed) / RAND_MAX * 20.0 - 10.0;  // 值在 [-10, 10]
        }
    }
}

void lu_in_situ_ver_final(double ** restrict A, int n) {
        /*
    1. Only write #parallel for in the outer loop
    2. Let all k loop work in this one parallel region
    3. Hence, the threads will be only create once,
       and the overhead of creating threads is reduced.
    4. The parallel loop in the middle (such as for (i = k+1; ...)) 
        can be directly divided into two parts using #pragma omp for, 
        so that each thread handles a part of the iterations.
    
    Effectively:
    Avoid frequent thread creation/destruction: 
    This is usually a large source of parallel overhead, 
    especially in small-scale loops.
    
    Make better use of data locality: 
    The same batch of threads in the same parallel region can access 
    specific cache/memory areas more stably without frequent 
    rescheduling to different cores.
    
    The iterations of the outer loop can be more flexibly scheduled: 
    sometimes all threads can perform step k at the same time, 
    and then step k+1, or other scheduling methods (such as work queues, etc.) 
    can be used to reduce load imbalance.

    Results:
    hame0798@vitsippa:~/LUproject/SecondVer$ ./situ 1 2048
    In-situ LU factorization time (Naive parallel): 4.903819 seconds
    In-situ LU factorization time (Optimized parallel): 4.931795 seconds
    hame0798@vitsippa:~/LUproject/SecondVer$ ./situ 4 2048
    In-situ LU factorization time (Naive parallel): 1.468962 seconds
    In-situ LU factorization time (Optimized parallel): 1.552530 seconds

    May not be useful since we haven't done any further optimization yet.
    */
    #pragma omp parallel
    {
        // 外层循环 k：每一步更新主元所在行及后续的子矩阵
        for (int k = 0; k < n; k++) {
            // 串行部分：对当前列 A[k+1:n][k] 进行除法更新
            #pragma omp single
            {
                double pivot = A[k][k];
                double temp = 1.0 / pivot;
                for (int i = k + 1; i < n; i++) {
                    A[i][k] *= temp;
                }
            }
            // 隐式 barrier 在 single 结束后同步所有线程

            // 内层消元更新：并行更新 A[k+1:n][k+1:n]
            // 采用静态循环调度（cyclic 分配，每个线程交替处理），保证数据局部性
            #pragma omp for schedule(static)
            for (int i = k + 1; i < n; i++) {
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
            }
            # pragma omp barrier
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




    parallel_initialize_matrix(A, n);


    //Save the original matrix for error calculation
    //double **A_orig = allocate_matrix(n);
    // for (int i = 0; i < n; i++){
    //     for (int j = 0; j < n; j++){
    //         A_orig[i][j] = A[i][j];
    //     }
    // }
    start_time = omp_get_wtime();
    lu_in_situ_ver_final(A, n);
    end_time = omp_get_wtime();
    lu_time = end_time - start_time;
    printf("In-situ LU factorization time (Optimized parallel): %f seconds\n", lu_time);

    //调用验证函数计算重构误差
    // double error = verify_lu(A_orig, A, n);
    // printf("\nFrobenius error: %e\n", error);
    
    
    
    // free_matrix(A_orig, n);
    free_matrix(A, n);

    return 0;
}