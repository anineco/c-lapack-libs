// getutime.c

#include <sys/resource.h>

#include "getutime.h"
#include "utils.h" // check

// getutime - get CPU time spent for user process in m seconds

long getutime(int diff)
{
  struct rusage usage;
  static long t1 = 0;
  long t2, tm;

#if defined(__linux__) && defined(_GNU_SOURCE)
# define who RUSAGE_THREAD
#else
# define who RUSAGE_SELF
#endif
  check(getrusage(who, &usage) == 0);
  t2 = usage.ru_utime.tv_sec * 1000 + (usage.ru_utime.tv_usec + 500) / 1000;
  tm = diff ? t2 - t1 : t2;
  t1 = t2;
  return (tm);
}

// __END__
