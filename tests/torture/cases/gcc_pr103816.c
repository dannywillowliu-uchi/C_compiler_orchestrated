/* Adapted from gcc.dg/torture/pr103816.c */

extern struct {
  unsigned char a;
  unsigned char b;
  unsigned char c;
  unsigned char d;
} g[];
void main() { g[0].b = (g[0].b & g[4].b) * g[2305843009213693952ULL].c; }
