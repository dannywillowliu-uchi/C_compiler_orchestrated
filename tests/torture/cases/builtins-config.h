/* Simplified builtins-config.h for torture test suite.
   Original from GCC testsuite does platform detection via <sys/types.h>.
   Our compiler targets a modern POSIX system, so we assume C99 runtime. */
#define HAVE_C99_RUNTIME
