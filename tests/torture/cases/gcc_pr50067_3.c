/* Adapted from gcc.dg/torture/pr50067-3.c */

extern void abort (void);
int a[6] = { 0, 0, 0, 0, 7, 0 };
static int *p = &a[4];

int
main ()
{
  int i;
  for (i = 0; i < 4; ++i)
    {
      a[i + 1] = a[i + 2] > i;
      *p &= ~1;
    }
  if (a[4] != 0)
    return 1;
  return 0;
}
