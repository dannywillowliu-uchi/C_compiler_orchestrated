/* Adapted from gcc.dg/torture/pr66375.c */

int a;
extern void abort (void);
int main ()
{
  int c = 0;
  for (; a < 13; ++a)
    c = (signed char)c - 11;
  if (c != 113)
    return 1;
  return 0;
}
