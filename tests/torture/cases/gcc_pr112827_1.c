/* Adapted from gcc.dg/torture/pr112827-1.c */

int a, b, c, d, e;
int main() {
  for (; c; c++) {
    for (a = 0; a < 2; a++)
      ;
    for (; b; b++) {
      e = d;
      d = a;
    }
  }
  return 0;
}
