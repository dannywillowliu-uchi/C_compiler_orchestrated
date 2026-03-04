/* Adapted from gcc.dg/torture/pr61602.c */
int a;
int *b = &a, **c = &b;
int
main ()
{
  int **d = &b;
  *d = 0;
}
