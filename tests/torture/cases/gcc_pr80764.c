/* Adapted from gcc.dg/torture/pr80764.c */

int b, d, f, h;
char e;
int fn1();
int main() { return fn1(); }
int fn1(int p1) {
    for (; d;)
      for (; e < 3;) {
	  for (; h;)
	    b = fn1(0);
	  return f;
      }
}
