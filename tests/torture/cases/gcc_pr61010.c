/* Adapted from gcc.dg/torture/pr61010.c */

int main (void)
{
  int a = 0;
  unsigned b = (a * 64 & 192) | 63U;
  return 0;
}
