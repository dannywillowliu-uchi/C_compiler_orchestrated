/* Adapted from gcc.dg/torture/fp-int-convert-float80-timode.c */
/* Test floating-point conversions.  __float80 type with TImode.  */
/* Origin: Joseph Myers <joseph@codesourcery.com> */

#include "fp-int-convert.h"

#define FLOAT80_MANT_DIG 64
#define FLOAT80_MAX_EXP 16384

int
main (void)
{
  TEST_I_F(TItype, UTItype, __float80, FLOAT80_MANT_DIG, FLOAT80_MAX_EXP);
  exit (0);
}
