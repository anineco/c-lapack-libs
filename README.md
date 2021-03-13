# c-lapack-libs
C coding example for linear algebra with LAPACK, CUDA, and vecLib

## Operating environment
- Linux + LAPACK
- Linux + CUDA
- macOS + LAPACK
- macOS + LAPACK (+Accelerate)
- macOS + vecLib
## Confirmed to work
- Linux: Fedora 33
- LAPACK: 3.9.0
- CUDA: 11.2
- macOS: 11.2.3
- vecLib: 1.11
## Compile and run
For example, to solve a simultaneous linear equation with 3000x3000 matrix
- LAPACK
```
make clean; make
./test_lusolv 3000
```
- CUDA
```
make clean; make USE_CUDA=1
./test_lusolv 3000
```
- vecLib
```
make clean; make USE_VECLIB=1
./test_lusolv 3000
```
