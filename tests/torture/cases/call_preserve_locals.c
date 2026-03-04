/* Adapted from gcc.dg/torture/call_preserve_locals.c -- tests calling convention */
/* Verify local variables survive across multiple function calls */

int consume(int a, int b, int c, int d) {
	return a + b + c + d;
}

int main(void) {
	int w = 100;
	int x = 200;
	int y = 300;
	int z = 400;

	int r1 = consume(1, 2, 3, 4);
	if (w != 100) return 1;
	if (x != 200) return 1;

	int r2 = consume(5, 6, 7, 8);
	if (y != 300) return 1;
	if (z != 400) return 1;

	int r3 = consume(w, x, y, z);
	if (r1 != 10) return 1;
	if (r2 != 26) return 1;
	if (r3 != 1000) return 1;

	return 0;
}
