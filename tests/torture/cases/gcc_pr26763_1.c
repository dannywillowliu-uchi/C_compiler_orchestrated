/* Adapted from gcc.dg/torture/pr26763-1.c */

extern void abort(void);

int try (int *a)
{
  return a + -1 > a;
}

int main(void)
{
  int bla[100];

  if (try (bla + 50))
    abort ();

  return 0;
}
