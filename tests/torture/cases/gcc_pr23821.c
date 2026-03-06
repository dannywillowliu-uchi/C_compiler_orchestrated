/* Adapted from gcc.dg/torture/pr23821.c */
/* At -O1 DOM threads a jump in a non-optimal way which leads to
   the bogus propagation.  */

int a[199];

extern void abort (void);

int
main ()
{
  int i, x;
  for (i = 0; i < 199; i++)
    a[i] = i;
  for (i = 0; i < 199; i++)
    {
      x = a[i];
      if (x != i)
	return 1;
    }
  return 0;
}

/* Verify that we do not propagate the equivalence x == i into the
   induction variable increment.  */

