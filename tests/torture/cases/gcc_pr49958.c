/* Adapted from gcc.dg/torture/pr49958.c */

extern void abort (void);
int foo (int i, int j, int o, int m) { return i*o + 1 + j*m > 1; }
int main()
{
  if (foo (- __INT_MAX__ - 1, -1, 1, 1))
    return 1;
  return 0;
}
