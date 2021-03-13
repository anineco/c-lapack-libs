CFLAGS = -O3 -Wall -Wextra
CPPFLAGS = -D_GNU_SOURCE

all: test_lusolv

ifeq ($(shell uname), Linux)
ifdef USE_CUDA

CU_VER = 11.2
MODULE = cusolver-$(CU_VER) cublas-$(CU_VER) cudart-$(CU_VER)
LDLIBS = $(shell pkg-config --libs $(MODULE)) -lm
lusolv.o: CPPFLAGS += -DUSE_CUDA
lusolv.o: CPPFLAGS += $(shell pkg-config --cflags $(MODULE))

else

MODULE = lapacke cblas
LDLIBS = $(shell pkg-config --libs $(MODULE)) -lm
lusolv.o: CPPFLAGS += $(shell pkg-config --cflags $(MODULE))

endif
endif # Linux

ifeq ($(shell uname), Darwin)
ifdef USE_VECLIB

LDLIBS = -framework Accelerate
lusolv.o: CPPFLAGS += -DUSE_VECLIB

else

LDLIBS = -L/opt/local/lib/lapack -llapacke -lcblas -lm
lusolv.o: CPPFLAGS += -I/opt/local/include/lapack

endif
endif # Darwin

test_lusolv: test_lusolv.o lusolv.o utils.o getutime.o
test_lusolv.o lusolv.o: lusolv.h utils.h getutime.h
lusolv.o: lusolv.h utils.h
utils.o: utils.h
getutime.o: getutime.h

clean:
	rm -f test_lusolv *.o
