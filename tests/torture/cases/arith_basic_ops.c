/* Adapted from gcc.dg/torture/arith_basic_ops.c -- tests arithmetic */

int main(void) {
	if (3 + 4 != 7) return 1;
	if (10 - 3 != 7) return 1;
	if (6 * 7 != 42) return 1;
	if (100 / 10 != 10) return 1;
	if (17 / 5 != 3) return 1;   /* integer division truncates */
	if (-7 + 3 != -4) return 1;
	if (0 * 999 != 0) return 1;
	if (1 * 42 != 42) return 1;
	if (-3 * -4 != 12) return 1;
	if (-15 / 3 != -5) return 1;
	return 0;
}
