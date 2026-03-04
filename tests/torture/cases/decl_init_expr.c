/* Adapted from gcc.dg/torture/decl_init_expr.c -- tests declarations */

int main(void) {
	int a = 1 + 2, b = 3 * 4;
	if (a != 3) return 1;
	if (b != 12) return 1;

	int c = a + b, d = c * 2;
	if (c != 15) return 1;
	if (d != 30) return 1;

	int e = (a > b) ? a : b;
	if (e != 12) return 1;

	return 0;
}
