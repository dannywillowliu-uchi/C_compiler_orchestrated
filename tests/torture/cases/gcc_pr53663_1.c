/* Adapted from gcc.dg/torture/pr53663-1.c */

extern void abort (void);

union u
{
  int i;
  _Bool b;
};

void f(union u * vp, union u v)
{
  *vp = v;
}

int main()
{
  union u v;
  union u v1;
  union u v2;

  v.i = 10;
  f(&v1, v);

  v.b = 0;
  f(&v2, v);
  if (v2.b != 0)
    return 1;
  return 0;
}
