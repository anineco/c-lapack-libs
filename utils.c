// utils.c

#include <stdio.h>
#include <stdlib.h> // exit, EXIT_FAILURE
#include <stdarg.h> // va_start, va_end
#include <string.h> // strerror
#include <errno.h> // errno

#include "utils.h"

void util_debug_print(const char *file, int line, const char *func, const char *fmt, ...)
{
  va_list ap;

  fprintf(stderr, "debug:%s:%d:%s:", file, line, func);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
}

int util_exit_print(const char *file, int line, const char *func, const char *fmt, ...)
{
  va_list ap;

  fprintf(stderr, "error:%s:%d:%s:", file, line, func);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, ":%s\n", strerror(errno));
  exit(EXIT_FAILURE);
  return 0;
}

// __END__
