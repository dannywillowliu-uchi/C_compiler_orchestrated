/* Adapted from gcc.dg/torture/bitint-83.c */
/* Derived from a test in gcc.dg/torture/bitint-16.c */

#include "../bitintext.h"

#define BASIC_TESTS \
  TEST(8)	    \
  TEST(16)	    \
  TEST(32)

#if __BITINT_MAXWIDTH__ >= 519
#define ALL_TESTS \
  BASIC_TESTS	  \
  TEST(64)	  \
  TEST(128)	  \
  TEST(256)	  \
  TEST(512)
#else
#define ALL_TESTS BASIC_TESTS
#endif

#define TEST(N) \
void						    \
test##N (unsigned _BitInt(N + 7) *t, _BitInt(N) x)  \
{						    \
  *t = -x;					    \
}
ALL_TESTS
#undef TEST

volatile int y = 0;

int
main (void)
{
#define TEST(N)	\
  {				\
    unsigned _BitInt(N + 7) t;	\
    _BitInt(N) x = y + N;	\
    test##N (&t, x);		\
    BEXTC (t);			\
  }
  ALL_TESTS
#undef TEST
}
