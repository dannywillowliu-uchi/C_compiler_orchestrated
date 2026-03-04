/* Adapted from gcc.dg/torture/pr42952.c */

extern void abort (void);

static int g[1];

static int * const p = &g[0];
static int * const q = &g[0];

int main(void)
{
  g[0] = 1;
  *p = 0;
  *p = *q;
  if (g[0] != 0)
    return 1;
  return 0;
}
