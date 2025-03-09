#include "LU_parallel.h"
#include <omp.h>

/* Parallel LU decomposition (Doolittle) */
void LUdecomposition_parallel(double **A, double **L, double **U, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    // Initialize matrices (串行初始化更高效，除非n极大)
    for (int i = 0; i < n; i++) {
        L[i][i] = 1.0;
        for (int j = 0; j < n; j++) {
            if (j != i) L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }

    // 主计算循环 - 仅并行计算密集部分
    for (int k = 0; k < n; k++) {
        // 计算U的第k行（可并行）
        #pragma omp parallel for schedule(dynamic, 16)
        for (int j = k; j < n; j++) {
            double sum = A[k][j];
            for (int m = 0; m < k; m++) {
                sum -= L[k][m] * U[m][j];
            }
            U[k][j] = sum;
        }

        // 预计算倒数避免重复计算
        const double diag_inv = 1.0 / U[k][k];
        
        // 计算L的第k列（可并行）
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = k + 1; i < n; i++) {
            double sum = A[i][k];
            for (int m = 0; m < k; m++) {
                sum -= L[i][m] * U[m][k];
            }
            L[i][k] = sum * diag_inv;
        }
    }
}

/* Parallel Cholesky decomposition */
int CholeskyDecomposition_parallel(double **A, double **L, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    // 初始化L为0（串行更高效）
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < n; j++) 
            L[i][j] = 0.0;

    for (int j = 0; j < n; j++) {
        // 计算对角线元素（带归约的并行）
        double sum = A[j][j];
        #pragma omp parallel for reduction(-:sum)
        for (int k = 0; k < j; k++)
            sum -= L[j][k] * L[j][k];
        
        if (sum <= 0.0) return -1;
        const double sqrt_sum = sqrt(sum);
        L[j][j] = sqrt_sum;
        const double inv_sqrt = 1.0 / sqrt_sum;

        // 计算列元素（可并行）
        #pragma omp parallel for
        for (int i = j + 1; i < n; i++) {
            double sum = A[i][j];
            for (int k = 0; k < j; k++)
                sum -= L[i][k] * L[j][k];
            L[i][j] = sum * inv_sqrt;
        }
    }
    return 0;
}

/* 带部分主元选择的并行LU分解 */
void PartialPivotingLU_parallel(double **A, double **L, double **U, int *P, int n, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // 初始化P和临时矩阵（串行）
    double **tempA = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        P[i] = i;
        tempA[i] = (double *)malloc(n * sizeof(double));
        L[i][i] = 1.0;
        for (int j = 0; j < n; j++) {
            tempA[i][j] = A[i][j];
            if (j != i) L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }

    for (int k = 0; k < n; k++) {
        // 主元选择（并行优化版本）
        int max_row = k;
        double max_val = fabs(tempA[k][k]);
        #pragma omp parallel for reduction(max:max_val, max_row)
        for (int i = k + 1; i < n; i++) {
            const double val = fabs(tempA[i][k]);
            if (val > max_val) {
                #pragma omp critical
                {
                    if (val > max_val) {
                        max_val = val;
                        max_row = i;
                    }
                }
            }
        }

        // 行交换（串行操作）
        if (max_row != k) {
            double *tmp = tempA[k];
            tempA[k] = tempA[max_row];
            tempA[max_row] = tmp;
            
            int tmp_p = P[k];
            P[k] = P[max_row];
            P[max_row] = tmp_p;
            
            for (int j = 0; j < k; j++) {
                double tmp_l = L[k][j];
                L[k][j] = L[max_row][j];
                L[max_row][j] = tmp_l;
            }
        }

        // 计算U的第k行（并行）
        #pragma omp parallel for
        for (int j = k; j < n; j++) {
            double sum = tempA[k][j];
            for (int m = 0; m < k; m++)
                sum -= L[k][m] * U[m][j];
            U[k][j] = sum;
        }

        // 计算L的第k列（并行）
        const double diag_inv = 1.0 / U[k][k];
        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            double sum = tempA[i][k];
            for (int m = 0; m < k; m++)
                sum -= L[i][m] * U[m][k];
            L[i][k] = sum * diag_inv;
        }
    }

    // 清理临时内存
    for (int i = 0; i < n; i++) free(tempA[i]);
    free(tempA);
}

/* 并行LDL^T分解 */
int LDLTDecomposition_parallel(double **A, double **L, double *D, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    // 对称性检查（优化版）
    int is_symmetric = 1;
    #pragma omp parallel for reduction(&&:is_symmetric) collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            if (A[i][j] != A[j][i]) 
                is_symmetric = 0;
    
    if (!is_symmetric) return -1;

    // 初始化L（串行）
    for (int i = 0; i < n; i++) {
        L[i][i] = 1.0;
        for (int j = 0; j < n; j++)
            if (j != i) L[i][j] = 0.0;
    }

    for (int j = 0; j < n; j++) {
        // 计算D[j]（并行归约）
        double sum = A[j][j];
        #pragma omp parallel for reduction(-:sum)
        for (int k = 0; k < j; k++)
            sum -= L[j][k] * L[j][k] * D[k];
        
        if (sum == 0.0) return -1;
        D[j] = sum;
        const double inv_d = 1.0 / sum;

        // 计算L的列（并行）
        #pragma omp parallel for
        for (int i = j + 1; i < n; i++) {
            double sum = A[i][j];
            for (int k = 0; k < j; k++)
                sum -= L[i][k] * L[j][k] * D[k];
            L[i][j] = sum * inv_d;
        }
    }
    return 0;
}