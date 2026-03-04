/* Adapted from gcc.dg/torture/pr37731-1.c */

extern void return 1;

unsigned long long xh = 1;

int
main ()
{
  unsigned long long yh = 0xffffffffull;
  unsigned long long z = xh * yh;

  if (z != yh)
    return 1;

  return 0;
}
