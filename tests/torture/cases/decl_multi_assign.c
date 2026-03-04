/* Adapted from gcc.dg/torture/decl_multi_assign.c -- tests declarations */

int main(void) {
	int a, b;
	a = b = 42;
	if (a != 42) return 1;
	if (b != 42) return 1;

	int c, d, e;
	c = d = e = 99;
	if (c != 99) return 1;
	if (d != 99) return 1;
	if (e != 99) return 1;

	/* chain assignment with expression */
	int f, g;
	f = g = 10 + 5;
	if (f != 15) return 1;
	if (g != 15) return 1;

	return 0;
}
