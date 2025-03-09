#include "LU_decomposition.h"

/* LU decomposition */
void LUdecomposition(double **A, double **L, double **U, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }
    for (int i = 0; i < n; i++) {
        // 计算 U 的第 i 行
        for (int j = i; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += L[i][k] * U[k][j];
            U[i][j] = A[i][j] - sum;
        }
        // 计算 L 的第 i 列
        for (int j = i + 1; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += L[j][k] * U[k][i];
            L[j][i] = (A[j][i] - sum) / U[i][i];
        }
    }
}

/* Cholesky 分解函数 */
int CholeskyDecomposition(double **A, double **L, int n) {
    // 初始化 L 为 0 矩阵
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            L[i][j] = 0.0;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double diff = A[i][i] - sum;
                if (diff <= 0.0) {
                    return -1;  // 非正定矩阵
                }
                L[i][j] = sqrt(diff);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return 0;
}

// 带部分选主元的LU分解，返回置换矩阵P（通过行交换数组表示）
void PartialPivotingLU(double **A, double **L, double **U, int *P, int n) {
    // 初始化P为行索引 [0, 1, ..., n-1]
    for (int i = 0; i < n; i++) P[i] = i;

    // 临时矩阵存储A，避免修改原矩阵
    double **tempA = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        tempA[i] = (double *)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            tempA[i][j] = A[i][j];
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }

    for (int k = 0; k < n; k++) {
        // 选主元：找到第k列中绝对值最大的行
        int max_row = k;
        double max_val = fabs(tempA[k][k]);
        for (int i = k + 1; i < n; i++) {
            if (fabs(tempA[i][k]) > max_val) {
                max_val = fabs(tempA[i][k]);
                max_row = i;
            }
        }

        // 交换行k和max_row
        if (max_row != k) {
            // 交换tempA的行
            double *tmp = tempA[k];
            tempA[k] = tempA[max_row];
            tempA[max_row] = tmp;

            // 交换P的记录
            int temp_p = P[k];
            P[k] = P[max_row];
            P[max_row] = temp_p;

            // 交换L中已计算的部分（仅交换前k-1列）
            for (int j = 0; j < k; j++) {
                double temp_l = L[k][j];
                L[k][j] = L[max_row][j];
                L[max_row][j] = temp_l;
            }
        }

        // 计算U的第k行和L的第k列
        for (int j = k; j < n; j++) {
            double sum = 0.0;
            for (int m = 0; m < k; m++) sum += L[k][m] * U[m][j];
            U[k][j] = tempA[k][j] - sum;
        }

        for (int i = k + 1; i < n; i++) {
            double sum = 0.0;
            for (int m = 0; m < k; m++) sum += L[i][m] * U[m][k];
            L[i][k] = (tempA[i][k] - sum) / U[k][k];
        }
    }

    // 释放临时矩阵
    for (int i = 0; i < n; i++) free(tempA[i]);
    free(tempA);
}



/* 打印矩阵 */
void printMatrix(double **M, int n, int m) {
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            printf("%8.4f ", M[i][j]);
        }
        printf("\n");
    }
}

/* 计算 A = L_true * L_true^T，要求 L_true 为下三角矩阵 */
void multiplyLowerTriangular(double **L_true, double **A, int n) {
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double sum = 0.0;
            int minIndex = (i < j) ? i : j;
            for (int k = 0; k <= minIndex; k++){
                sum += L_true[i][k] * L_true[j][k];
            }
            A[i][j] = sum;
        }
    }
}