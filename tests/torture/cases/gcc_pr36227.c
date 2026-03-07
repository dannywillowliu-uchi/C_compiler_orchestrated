/* Adapted from gcc.dg/torture/pr36227.c */

extern void abort (void);
int main()
{
  int i = 1;
  int *p = &i;
  uintptr_t iptr;

  iptr = (uintptr_t)p - (uintptr_t)&iptr;
  p = (int *)((uintptr_t)&iptr + iptr);
  if (*p != 1)
    abort ();
  return 0;
}

