/* Adapted from gcc.dg/torture/bitint-29.c */
/* PR c/102989 */

#if __BITINT_MAXWIDTH__ >= 119
#include "../../c-c++-common/torture/builtin-arith-overflow-1.h"

#define U(s, op) op
TESTS (_BitInt(119), (-332306998946228968225951765070086143wb - 1), 332306998946228968225951765070086143wb)

#undef T
#define T(n, t1, t2, tr, v1, v2, vr, b, o) t##n##b ();
#endif

int
main ()
{
#if __BITINT_MAXWIDTH__ >= 119
  TESTS (_BitInt(119), (-332306998946228968225951765070086143wb - 1), 332306998946228968225951765070086143wb)
#endif
  return 0;
}
