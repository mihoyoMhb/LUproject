// These are the additional parallel test functions to add to Test_LU.c

// Test parallel LU decomposition
double testLUdecomposition_parallel(double **A, int n, int num_threads) {
    printf("\n--- Testing Parallel LU decomposition (%d threads) ---\n", num_threads);
    
    // For original algorithm (baseline)
    double **L = allocateMatrix(n);
    double **U = allocateMatrix(n);
    double **LU = allocateMatrix(n);
    
    // Measure time for original algorithm
    clock_t start = clock();
    LUdecomposition(A, L, U, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Verify result: A = L*U
    multiplyMatrices(L, U, LU, n);
    double error_orig = matrixError(A, LU, n);
    
    printf("Original algorithm:\n");
    printf("  Time taken: %f seconds\n", time_orig);
    printf("  Reconstruction error: %e\n", error_orig);
    
    // For parallel algorithm
    double **L_par = allocateMatrix(n);
    double **U_par = allocateMatrix(n);
    double **LU_par = allocateMatrix(n);
    
    // Measure time for parallel algorithm
    start = clock();
    LUdecomposition_parallel(A, L_par, U_par, n, num_threads);
    end = clock();
    double time_par = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Verify result: A = L*U
    multiplyMatrices(L_par, U_par, LU_par, n);
    double error_par = matrixError(A, LU_par, n);
    
    double speedup = time_orig / time_par;
    
    printf("Parallel algorithm (%d threads):\n", num_threads);
    printf("  Time taken: %f seconds\n", time_par);
    printf("  Reconstruction error: %e\n", error_par);
    printf("  Speedup: %.2f times\n", speedup);
    printf("  Parallel efficiency: %.2f%%\n", (speedup / num_threads) * 100.0);
    
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("L matrix (original):\n");
        printMatrix(L, n, n);
        printf("U matrix (original):\n");
        printMatrix(U, n, n);
        printf("L matrix (parallel):\n");
        printMatrix(L_par, n, n);
        printf("U matrix (parallel):\n");
        printMatrix(U_par, n, n);
    }
    
    freeMatrix(L, n);
    freeMatrix(U, n);
    freeMatrix(LU, n);
    freeMatrix(L_par, n);
    freeMatrix(U_par, n);
    freeMatrix(LU_par, n);
    
    return speedup;
}

// Test parallel Cholesky decomposition
double testCholeskyDecomposition_parallel(double **A, int n, int num_threads) {
    printf("\n--- Testing Parallel Cholesky decomposition (%d threads) ---\n", num_threads);
    
    // For original algorithm
    double **L = allocateMatrix(n);
    double **LLT = allocateMatrix(n);
    
    // Measure time for original algorithm
    clock_t start = clock();
    int result = CholeskyDecomposition(A, L, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result == -1) {
        printf("Original algorithm: Cholesky decomposition failed - Matrix is not positive definite\n");
        freeMatrix(L, n);
        freeMatrix(LLT, n);
        return -1.0;
    }
    
    // Verify result: A = L*L^T
    multiplyLowerTriangular(L, LLT, n);
    double error_orig = matrixError(A, LLT, n);
    
    printf("Original algorithm:\n");
    printf("  Time taken: %f seconds\n", time_orig);
    printf("  Reconstruction error: %e\n", error_orig);
    
    // For parallel algorithm
    double **L_par = allocateMatrix(n);
    double **LLT_par = allocateMatrix(n);
    
    // Measure time for parallel algorithm
    start = clock();
    int result_par = CholeskyDecomposition_parallel(A, L_par, n, num_threads);
    end = clock();
    double time_par = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (result_par == -1) {
        printf("Parallel algorithm: Cholesky decomposition failed - Matrix is not positive definite\n");
        freeMatrix(L, n);
        freeMatrix(LLT, n);
        freeMatrix(L_par, n);
        freeMatrix(LLT_par, n);
        return -1.0;
    }
    
    // Verify result: A = L*L^T
    multiplyLowerTriangular(L_par, LLT_par, n);
    double error_par = matrixError(A, LLT_par, n);
    
    double speedup = time_orig / time_par;
    
    printf("Parallel algorithm (%d threads):\n", num_threads);
    printf("  Time taken: %f seconds\n", time_par);
    printf("  Reconstruction error: %e\n", error_par);
    printf("  Speedup: %.2f times\n", speedup);
    printf("  Parallel efficiency: %.2f%%\n", (speedup / num_threads) * 100.0);
    
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("L matrix (original):\n");
        printMatrix(L, n, n);
        printf("L matrix (parallel):\n");
        printMatrix(L_par, n, n);
    }
    
    freeMatrix(L, n);
    freeMatrix(LLT, n);
    freeMatrix(L_par, n);
    freeMatrix(LLT_par, n);
    
    return speedup;
}

// Test parallel LU decomposition with partial pivoting
double testPartialPivotingLU_parallel(double **A, int n, int num_threads) {
    printf("\n--- Testing Parallel LU decomposition with Partial Pivoting (%d threads) ---\n", num_threads);
    
    // For original algorithm
    double **L = allocateMatrix(n);
    double **U = allocateMatrix(n);
    double **PA = allocateMatrix(n);
    double **LU = allocateMatrix(n);
    int *P = (int *)malloc(n * sizeof(int));
    
    // Make a copy of A to avoid modifying it
    double **A_copy = allocateMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_copy[i][j] = A[i][j];
        }
    }
    
    // Measure time for original algorithm
    clock_t start = clock();
    PartialPivotingLU(A_copy, L, U, P, n);
    clock_t end = clock();
    double time_orig = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Apply permutation to A to get PA
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            PA[i][j] = A[P[i]][j];
        }
    }
    
    // Verify result: PA = L*U
    multiplyMatrices(L, U, LU, n);
    double error_orig = matrixError(PA, LU, n);
    
    printf("Original algorithm:\n");
    printf("  Time taken: %f seconds\n", time_orig);
    printf("  Reconstruction error: %e\n", error_orig);
    
    // For parallel algorithm
    double **L_par = allocateMatrix(n);
    double **U_par = allocateMatrix(n);
    double **PA_par = allocateMatrix(n);
    double **LU_par = allocateMatrix(n);
    int *P_par = (int *)malloc(n * sizeof(int));
    
    // Make another copy of A
    double **A_copy_par = allocateMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_copy_par[i][j] = A[i][j];
        }
    }
    
    // Measure time for parallel algorithm
    start = clock();
    PartialPivotingLU_parallel(A_copy_par, L_par, U_par, P_par, n, num_threads);
    end = clock();
    double time_par = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Apply permutation to A to get PA
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            PA_par[i][j] = A[P_par[i]][j];
        }
    }
    
    // Verify result: PA = L*U
    multiplyMatrices(L_par, U_par, LU_par, n);
    double error_par = matrixError(PA_par, LU_par, n);
    
    double speedup = time_orig / time_par;
    
    printf("Parallel algorithm (%d threads):\n", num_threads);
    printf("  Time taken: %f seconds\n", time_par);
    printf("  Reconstruction error: %e\n", error_par);
    printf("  Speedup: %.2f times\n", speedup);
    printf("  Parallel efficiency: %.2f%%\n", (speedup / num_threads) * 100.0);
    
    if (n <= 5) {
        printf("Original matrix A:\n");
        printMatrix(A, n, n);
        printf("Permutation P (original): ");
        for (int i = 0; i < n; i++) printf("%d ", P[i]);
        printf("\nPermutation P (parallel): ");
        for (int i = 0; i < n; i++) printf("%d ", P_par[i]);
        printf("\n");
        printf("L matrix (original):\n");
        printMatrix(L, n, n);
        printf("U matrix (original):\n");
        printMatrix(U, n, n);
        printf("L matrix (parallel):\n");
        printMatrix(L_par, n, n);
        printf("U matrix (parallel):\n");
        printMatrix(U_par, n, n);
    }
    
    free(P);
    free(P_par);
    freeMatrix(L, n);
    freeMatrix(U, n);
    freeMatrix(PA, n);
    freeMatrix(LU, n);
    freeMatrix(A_copy, n);
    freeMatrix(L_par, n);
    freeMatrix(U_par, n);
    freeMatrix(PA_par, n);
    freeMatrix(LU_par, n);
    freeMatrix(A_copy_par, n);
    
    return speedup;
}

// Test parallel LDL^T decomposition
double testLDLTDecomposition_parallel(double **A, int n, int num_threads) {
    printf("\n--- Testing Parallel LDL^T decomposition (%d threads) ---\n", num_threads);
    
    // For original algorithm
    double **L = allocateMatrix(n);
    double *D = (double *)malloc(n * sizeof(double));
    double **LDLT