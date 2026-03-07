/* Adapted from gcc.dg/torture/pr116795-1.c */

volatile int a, b;
int c;
int main() {
  unsigned e = 0;
  for (; e < 2; e++) {
    a && b;
    if (c)
      e = -(c ^ e);
  }
  return 0;
}
