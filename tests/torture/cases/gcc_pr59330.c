/* Adapted from gcc.dg/torture/pr59330.c */

void free(void *ptr)
{
}

void *foo(void)
{
  return 0;
}

int main(void)
{
  void *p = foo();
  free(p);
  return 0;
}
