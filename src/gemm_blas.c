
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cblas.h>

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    int n = 512;

    if (argc > 1)
        n = atoi(argv[1]);

    size_t elements = (size_t)n * n;

    float *A = malloc(elements * sizeof(float));
    float *B = malloc(elements * sizeof(float));
    float *C = calloc(elements, sizeof(float));

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    srand(42);

    for (size_t i = 0; i < elements; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    // Warm-up
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n, 1.0f, A, n, B, n, 0.0f, C, n
    );

    double times[5];

    for (int run = 0; run < 5; run++) {
        double start = get_time();

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, n, n, 1.0f, A, n, B, n, 0.0f, C, n
        );

        times[run] = (get_time() - start) * 1000.0;
    }

    double mean = 0.0;

    for (int i = 0; i < 5; i++) {
        mean += times[i];
    }

    mean /= 5.0;

    double variance = 0.0;

    for (int i = 0; i < 5; i++) {
        double difference = times[i] - mean;
        variance += difference * difference;
    }

    double sd = sqrt(variance / 5.0);
    double gflops = (2.0 * n * n * n) / (mean * 1e6);

    printf("\nOpenBLAS GEMM\n");
    printf("Matrix size: %d x %d\n", n, n);

    for (int i = 0; i < 5; i++)
        printf("Run %d: %.3f ms\n", i + 1, times[i]);

    printf("Mean: %.3f ms\n", mean);
    printf("Standard deviation: %.3f ms\n", sd);
    printf("GFLOP/s: %.4f\n", gflops);

    free(A);
    free(B);
    free(C);

    return 0;
}
