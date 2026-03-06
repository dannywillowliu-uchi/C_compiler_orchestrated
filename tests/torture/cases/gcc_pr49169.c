/* Adapted from gcc.dg/torture/pr49169.c */


int
main (void)
{
  void *p = main;
  if ((intptr_t) p & 1)
    abort ();
  return 0;
}

