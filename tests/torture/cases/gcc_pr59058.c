/* Adapted from gcc.dg/torture/pr59058.c */

extern void abort (void);

short b = 0;

int
main ()
{
  int c = 0;
l1:
  b++;
  c |= b;
  if (b)
    goto l1;
  if (c != -1)
    abort ();
  return 0;
}
