/* Adapted from gcc.dg/torture/bitint-32.c */
/* PR c/102989 */

#if __BITINT_MAXWIDTH__ >= 135
#include "../../c-c++-common/torture/builtin-arith-overflow-1.h"

#define U(s, op) op
TESTS (_BitInt(135), (-21778071482940061661655974875633165533183wb - 1), 21778071482940061661655974875633165533183wb)

#undef T
#define T(n, t1, t2, tr, v1, v2, vr, b, o) t##n##b ();
#endif

int
main ()
{
#if __BITINT_MAXWIDTH__ >= 135
  TESTS (_BitInt(135), (-21778071482940061661655974875633165533183wb - 1), 21778071482940061661655974875633165533183wb)
#endif
  return 0;
}
