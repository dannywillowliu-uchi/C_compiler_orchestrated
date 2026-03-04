/* Adapted from gcc.dg/torture/pr123887-2.c */

[[gnu::noipa]]
int f(int a, int b, int *x)
{
  return (a ? *x : 0) != (b ? *x : 0);
}
int main()
{
  f(0, 0, 0);
  return 0;
}
