/* Adapted from gcc.dg/torture/pr104279.c */
/* PR tree-optimization/104279 */

unsigned a, b;

int
main ()
{
  b = ~(0 || ~0);
  a = ~b / ~a;
  return 0;
}
