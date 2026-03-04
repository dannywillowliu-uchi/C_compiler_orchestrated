/* Adapted from gcc.dg/torture/pr54027.c */

int main (void)
{
  int x = 1;
  while (x)
    x <<= 1;
  return x;
}
