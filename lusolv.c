// lusolv.c

#include <stdio.h>
#include <string.h> // memcpy
#if defined(USE_CUDA)
# include <cuda_runtime.h>
# include <cublas_v2.h>
# include <cusolverDn.h>
#elif defined(USE_VECLIB)
# include <Accelerate/Accelerate.h>
#else
# include <cblas.h>
# include <lapacke.h>
#endif

#include "lusolv.h"
#include "utils.h"

#if defined(USE_CUDA)
static cusolverDnHandle_t handle;
static float *d_a;
static float *d_x;
static float *d_b;
static float *d_work;
static int *d_ipiv;
static int *d_info;
#else
static int *ipiv;
#endif

void luinit(int n, int m)
{
#if defined(USE_CUDA)
  int n_work;

  check(cusolverDnCreate(&handle) == CUSOLVER_STATUS_SUCCESS);
  check(cudaMalloc((void **)&d_a, sizeof (*d_a) * n * n) == cudaSuccess);
  check(cudaMalloc((void **)&d_x, sizeof (*d_x) * n * m) == cudaSuccess);
  check(cudaMalloc((void **)&d_b, sizeof (*d_b) * n * m) == cudaSuccess);
  check(cusolverDnSgetrf_bufferSize(handle, n, n, d_a, n, &n_work) == CUSOLVER_STATUS_SUCCESS);
  check(cudaMalloc((void **)&d_work, sizeof (*d_work) * n_work) == cudaSuccess);
  check(cudaMalloc((void **)&d_ipiv, sizeof (*d_ipiv) * n) == cudaSuccess);
  check(cudaMalloc((void **)&d_info, sizeof (*d_info)) == cudaSuccess);
#else
  (void)m; // unused
  check(ipiv = malloc(sizeof (*ipiv) * n));
#endif
}

// x := a
void lucopy(int n, int m, float *a, float *x)
{
  memcpy(x, a, n * m * sizeof (*x));
}

// b := a * x
void lugemm(int n, int m, float *a, float *x, float *b)
{
  float alpha = 1.0, beta = 0.0;
#if defined(USE_CUDA)
  cublasHandle_t handle;

  check(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
  check(cublasSetMatrix(n, n, sizeof (*a), a, n, d_a, n) == CUBLAS_STATUS_SUCCESS);
  check(cublasSetMatrix(n, m, sizeof (*x), x, n, d_x, n) == CUBLAS_STATUS_SUCCESS);
  check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, n, &alpha, d_a, n, d_x, n, &beta, d_b, n)  == CUBLAS_STATUS_SUCCESS);
  check(cublasGetMatrix(n, m, sizeof (*b), d_b, n, b, n) == CUBLAS_STATUS_SUCCESS);
  check(cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS);
#else
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
    n, m, n, alpha, a, n, x, n, beta, b, n);
#endif
}

void ludcmp(int n, float *a)
{
#if defined(USE_CUDA)
  int info;

  check(cudaMemcpy(d_a, a, sizeof (*d_a) * n * n, cudaMemcpyHostToDevice) == cudaSuccess);
  check(cusolverDnSgetrf(handle, n, n, d_a, n, d_work, d_ipiv, d_info) == CUSOLVER_STATUS_SUCCESS);
  check(cudaMemcpy(&info, d_info, sizeof (*d_info), cudaMemcpyDeviceToHost) == cudaSuccess);
  check(info == 0, "info=%d", info);
#elif defined(USE_VECLIB)
  int info;

  sgetrf_(&n, &n, a, &n, ipiv, &info);
  check(info == 0, "info=%d", info);
#else
  lapack_int info;

  info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, a, n, ipiv);
  check(info == 0, "info=%d", info);
#endif
}

void lusolv(int n, int m, float *a, float *x)
{
#if defined(USE_CUDA)
  int info;
  (void)a; // unused

  check(cudaMemcpy(d_x, x, sizeof (*d_x) * n * m, cudaMemcpyHostToDevice) == cudaSuccess);
  check(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, m, d_a, n, d_ipiv, d_x, n, d_info) == CUSOLVER_STATUS_SUCCESS);
  check(cudaMemcpy(x, d_x, sizeof (*d_x) * n * m, cudaMemcpyDeviceToHost) == cudaSuccess);
  check(cudaMemcpy(&info, d_info, sizeof (*d_info), cudaMemcpyDeviceToHost) == cudaSuccess);
  check(info == 0, "info=%d", info);
#elif defined(USE_VECLIB)
  int info;

  sgetrs_("N", &n, &m, a, &n, ipiv, x, &n, &info);
  check(info == 0, "info=%d", info);
#else
  lapack_int info;

  info = LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, m, a, n, ipiv, x, n);
  check(info == 0, "info=%d", info);
#endif
}

void luterm(void)
{
#if defined(USE_CUDA)
  check(cudaFree(d_a) == cudaSuccess);
  check(cudaFree(d_x) == cudaSuccess);
  check(cudaFree(d_work) == cudaSuccess);
  check(cudaFree(d_ipiv) == cudaSuccess);
  check(cudaFree(d_info) == cudaSuccess);
  check(cusolverDnDestroy(handle) == CUSOLVER_STATUS_SUCCESS);
#else
  free(ipiv);
#endif
}

// end of lusolv.c
