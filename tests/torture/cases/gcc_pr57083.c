/* Adapted from gcc.dg/torture/pr57083.c */
/* PR tree-optimization/57083 */

extern void abort (void);
short x = 1;
int y = 0;

int
main ()
{
  unsigned t = (0x7fff8001U - x) << (y == 0);
  if (t != 0xffff0000U)
    return 1;
  return 0;
}
