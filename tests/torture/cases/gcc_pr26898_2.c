/* Adapted from gcc.dg/torture/pr26898-2.c */


int a = 0, b = INT_MAX - 1;
extern void abort(void);
int main()
{
  if (a - 1 > b + 1)
    return 1;
  return 0;
}
