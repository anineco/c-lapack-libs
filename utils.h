// utils.h

#if !(defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L)
# error compiler conforming to the 1999 C Standard or later is required.
#endif

void util_debug_print(const char *file, int line, const char *func, const char *fmt, ...);
int util_exit_print(const char *file, int line, const char *func, const char *fmt, ...);

#ifdef DEBUG
# define debug(...)       util_debug_print(__FILE__, __LINE__, __func__, ""__VA_ARGS__)
# define debugvar(fmt, e) util_debug_print(__FILE__, __LINE__, __func__, #e "=" fmt, e)
#else
# define debug(...)       ((void)0)
# define debugvar(fmt, e) ((void)0)
#endif

#define die(...)          ((void)util_exit_print(__FILE__, __LINE__, __func__, ""__VA_ARGS__))
#define check(e, ...)     ((void)((e) || util_exit_print(__FILE__, __LINE__, __func__, ""__VA_ARGS__)))

// __END__
