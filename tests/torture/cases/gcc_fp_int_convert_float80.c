/* Adapted from gcc.dg/torture/fp-int-convert-float80.c */
/* Test floating-point conversions.  __float80 type.  */
/* Origin: Joseph Myers <joseph@codesourcery.com> */

#include "fp-int-convert.h"

#define FLOAT80_MANT_DIG 64
#define FLOAT80_MAX_EXP 16384

int
main (void)
{
  TEST_I_F(signed char, unsigned char, __float80, FLOAT80_MANT_DIG, FLOAT80_MAX_EXP);
  TEST_I_F(signed short, unsigned short, __float80, FLOAT80_MANT_DIG, FLOAT80_MAX_EXP);
  TEST_I_F(signed int, unsigned int, __float80, FLOAT80_MANT_DIG, FLOAT80_MAX_EXP);
  TEST_I_F(signed long, unsigned long, __float80, FLOAT80_MANT_DIG, FLOAT80_MAX_EXP);
  TEST_I_F(signed long long, unsigned long long, __float80, FLOAT80_MANT_DIG, FLOAT80_MAX_EXP);
  return 0;
}
