/* Adapted from gcc.dg/torture/20141202-1.c */

extern void abort (void);

int foo (int x)
{
  return (x / 2) / ((-__INT_MAX__ - 1) / -2);
}

int main()
{
  if (foo (- __INT_MAX__ - 1) != -1)
    return 1;
  return 0;
}
