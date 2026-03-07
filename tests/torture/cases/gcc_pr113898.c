/* Adapted from gcc.dg/torture/pr113898.c */

int a, d;
unsigned **b;
long c, f;
long e[2][1];
void g() {
  int h = 0;
  for (; h < 2; h++) {
    e[h][d + **b + a] = c;
    if (f)
      for (;;)
        ;
  }
}
void main() {}
