/* Adapted from gcc.dg/torture/pr113895-2.c */

extern void d(int);
int a[2][4], b;
int main() {
  while (b) {
    int c;
    d(a[b][c]);
    for (c = 0; c < 7; c++)
      ;
  }
  return 0;
}
