// lusolv.h

void luinit(int n, int m);
void lucopy(int n, int m, float *a, float *x);
void lugemm(int n, int m, float *a, float *x, float *b);
void ludcmp(int n, float *a);
void lusolv(int n, int m, float *a, float *x);
void luterm(void);

#define IDX2C(i, j, ld) ((i) + (j) * (ld))

// __END__
