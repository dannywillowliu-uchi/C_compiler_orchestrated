/* Adapted from gcc.dg/torture/20110719-1.c */
extern void abort (void);
int i;
int main()
{
  int b = i != 0;
  int c = ~b;
  if (c != -1)
    return 1;
  return 0;
}
