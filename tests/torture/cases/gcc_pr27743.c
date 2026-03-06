/* Adapted from gcc.dg/torture/pr27743.c */

extern void abort(void);

int32_t bar (int32_t a)
{
  return ((uint32_t) ((a) >> 2)) >> 15;
}

int main()
{
  if (bar (0xffff3000) != 0x1ffff)
    abort ();
  return 0;
}
