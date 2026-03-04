/* Adapted from gcc.dg/torture/pr53663-2.c */

extern void abort (void);

union u
{
  int i;
  short f;
} v;

short foo (short *f)
{
  *f = 1;
  v.i = 0;
  v.f = 0;
  return *f;
}

int main()
{
  if (foo (&v.f) != 0)
    return 1;
  return 0;
}
