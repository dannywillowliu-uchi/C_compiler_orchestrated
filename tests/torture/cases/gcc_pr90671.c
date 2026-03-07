/* Adapted from gcc.dg/torture/pr90671.c */
/* PR tree-optimization/90671 */

int a;

int
main ()
{
  int b, c;
  for (c = 0; c < 2; c++)
    while (a)
      if (b)
	break;
  return 0;
}
