/* Adapted from gcc.dg/torture/pr105786.c */

void sink(const char*);
static const char *a = "ab\0cd";
int main()
{
  const char *b = a;
  for (int i = 0; i < 2; ++i)
    while (*b++)
      ;
  sink(b);
  return 0;
}
