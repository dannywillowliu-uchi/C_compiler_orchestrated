/* Adapted from gcc.dg/torture/pr110766.c */

int a, b, c, e;
short d, f;
int g(int h) { return h > a ? h : h << a; }
int main() {
  while (e) {
    b = 0;
    for (; b < 3; b++)
      if (c) {
        e = g(1);
        f = e | d;
      }
    d = 0;
  }
  return 0;
}
