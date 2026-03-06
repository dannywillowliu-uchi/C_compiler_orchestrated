/* Adapted from gcc.dg/torture/pr37731-2.c */

extern void abort(void);

long long xh = 1;

int
main ()
{
  long long yh = 0xffffffffll;
  long long z = xh * yh;

  if (z != yh)
    return 1;

  return 0;
}
