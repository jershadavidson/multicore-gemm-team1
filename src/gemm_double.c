
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

#define N 1024
#define RUNS 5

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main() {
    int n = N;

    double *A = malloc((size_t)n * n * sizeof(double));
    double *B = malloc((size_t)n * n * sizeof(double));
    double *C = malloc((size_t)n * n * sizeof(double));
    double *R = malloc((size_t)n * n * sizeof(double));

    if (!A || !B || !C || !R) {
        printf("Memory allocation failed\n");
        return 1;
    }

    srand(42);

    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    // OpenBLAS double-precision reference
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n, 1.0, A, n, B, n, 0.0, R, n
    );

    // Warm-up
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n, 1.0, A, n, B, n, 0.0, C, n
    );

    double times[RUNS];

    for (int run = 0; run < RUNS; run++) {
        double start = get_time();

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, n, n, 1.0, A, n, B, n, 0.0, C, n
        );

        double end = get_time();
        times[run] = (end - start) * 1000.0;

        printf("Run %d: %.3f ms\n", run + 1, times[run]);
    }

    double sum = 0.0;
    for (int i = 0; i < RUNS; i++)
        sum += times[i];

    double mean = sum / RUNS;
    double variance = 0.0;

    for (int i = 0; i < RUNS; i++) {
        double diff = times[i] - mean;
        variance += diff * diff;
    }

    double sd = sqrt(variance / (RUNS - 1));
    double max_error = 0.0;
    double max_reference = 0.0;

    for (int i = 0; i < n * n; i++) {
        double error = fabs(C[i] - R[i]);
        if (error > max_error) max_error = error;
        if (fabs(R[i]) > max_reference)
            max_reference = fabs(R[i]);
    }

    double relative_error =
        max_reference > 0.0 ? max_error / max_reference : 0.0;

    double gflops = (2.0 * n * n * n) / (mean * 1e6);

    printf("\nPrecision: Double\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Mean: %.3f ms\n", mean);
    printf("Standard deviation: %.3f ms\n", sd);
    printf("GFLOP/s: %.4f\n", gflops);
    printf("Relative max-norm error: %.12e\n", relative_error);

    free(A);
    free(B);
    free(C);
    free(R);

    return 0;
}
