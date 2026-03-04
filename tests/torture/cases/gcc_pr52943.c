/* Adapted from gcc.dg/torture/pr52943.c */

extern void abort (void);
int a[] = { 0, 0, 0, 6 };

int b;
int
main ()
{
  for (;;)
    {
      b = 3;
      for (; b; b -= 1)
	a[b] = a[3] > 1;
      break;
    }
  if (a[1] != 0)
    return 1;
  return 0;
}
