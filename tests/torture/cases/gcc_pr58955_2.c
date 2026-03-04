/* Adapted from gcc.dg/torture/pr58955-2.c */

extern void abort (void);

int a, b[10];

int
main ()
{
  for (; a < 2; a++)
    {
      b[a] = 1;
      b[a + 1] = 0;
    }
  if (b[1] != 1)
    return 1;
  return 0;
}
