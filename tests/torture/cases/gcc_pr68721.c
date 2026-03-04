/* Adapted from gcc.dg/torture/pr68721.c */

extern void abort (void);

int a, b, c, *d, **e = &d;

int *
fn1 ()
{
  for (;;)
    {
      for (; a;)
	if (b)
	  return 1;
      break;
    }
  for (; c;)
    ;
  return &a;
}

int
main ()
{
  *e = fn1 ();

  if (!d)
    return 1;

  return 0;
}
