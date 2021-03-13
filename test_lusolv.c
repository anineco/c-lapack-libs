// test_lusolv.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lusolv.h"
#include "utils.h"
#include "getutime.h"

void print_mat(int m, int n, float *a)
{
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%10.6f%c", a[IDX2C(i, j, m)], (j < n - 1) ? ' ' : '\n');
    }
  }
}

void set_rndmat(float *v, int n, float lo, float up)
{
  drand48();
  for (int i = 0; i < n; i++) {
    v[i] = lo + (up - lo) * drand48();
  }
}

void print_esterr(int n, float *a, float *b)
{
  double emax = 0;
  double sum = 0;
  for (int i = 0; i < n; i++) {
    double w = fabs(a[i] - b[i]);
    emax = fmax(w, emax);
    sum += w * w;
  }
  printf(" max|X-true(X)|=%g\n", emax);
  printf("RMSE(X,true(X))=%g\n", sqrt(sum / n));
}

int main(int argc, const char **argv)
{
  int n, m;
  float *a;   // a[n * n]
  float *b;   // b[n * m]
  float *x;   // x[n * m]
  float *x0;  // x0[n * m]

  n = (argc > 1) ? atoi(argv[1]) : 10;
  m = n;

  luinit(n, m);

  check(a = calloc(n * n, sizeof (*a)));
  check(b = calloc(n * m, sizeof (*b)));
  check(x = calloc(n * m, sizeof (*x)));
  check(x0 = calloc(n * m, sizeof (*x0))); // true X

  set_rndmat(a, n * n, -1, 1);
  set_rndmat(x0, n * m, -1, 1);

  getutime(0);
  lugemm(n, m, a, x0, b); // b := a * x0
  printf("sgemm:%ld[ms]\n", getutime(1));

  if (n <= 10) {
    printf("===== matrix a\n");
    print_mat(n, n, a);

    printf("===== matrix x0\n");
    print_mat(n, m, x0);

    printf("===== matrix b=a*x0\n");
    print_mat(n, m, b);
  }

  getutime(0);
  ludcmp(n, a);
  printf("ludcmp:%ld[ms]\n", getutime(1));

  lucopy(n, m, b, x); // x := b
  getutime(0);
  lusolv(n, m, a, x); // solve x for a * x = b
  printf("lusolv:%ld[ms]\n", getutime(1));
  if (n <= 10) {
    printf("===== matrix x\n");
    print_mat(n, m, x);
  }
  print_esterr(n * m, x, x0);

  luterm();
  free(a);
  free(b);
  free(x);
  return 0;
}

// __END__
