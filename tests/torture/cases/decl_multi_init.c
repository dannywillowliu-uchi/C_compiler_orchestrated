/* Adapted from gcc.dg/torture/decl_multi_init.c -- tests declarations */

int main(void) {
	int a = 1, b = 2, c = 3;
	if (a != 1) return 1;
	if (b != 2) return 1;
	if (c != 3) return 1;

	int x = 100, y = 200;
	if (x + y != 300) return 1;

	return 0;
}
