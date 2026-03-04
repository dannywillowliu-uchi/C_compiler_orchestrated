/* Adapted from gcc.dg/torture/decl_multi_int.c -- tests declarations */

int main(void) {
	int a, b, c;
	a = 10;
	b = 20;
	c = 30;
	if (a != 10) return 1;
	if (b != 20) return 1;
	if (c != 30) return 1;
	if (a + b + c != 60) return 1;
	return 0;
}
