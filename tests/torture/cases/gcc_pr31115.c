/* Adapted from gcc.dg/torture/pr31115.c */

extern void exit(int);
extern void abort(void);
void foo (int e1)
{
  if (e1 < 0)
    {
      e1 = -e1;
      if (e1 >>= 4)
        {
          if (e1 >= 1 << 5)
            return 0;
        }
    }
}

int main()
{
  foo(-(1<<9));
  return 1;
}
