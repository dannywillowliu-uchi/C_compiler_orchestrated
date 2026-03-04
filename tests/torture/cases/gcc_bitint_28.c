/* Adapted from gcc.dg/torture/bitint-28.c */
/* PR c/102989 */

#include "../../c-c++-common/torture/builtin-arith-overflow-1.h"

#define U(s, op) op
TESTS (_BitInt(19), (-262143wb - 1), 262143wb)

#undef T
#define T(n, t1, t2, tr, v1, v2, vr, b, o) t##n##b ();

int
main ()
{
  TESTS (_BitInt(19), (-262143wb - 1), 262143wb)
  return 0;
}
