/* Adapted from gcc.dg/torture/arith_precedence.c -- tests arithmetic */

int main(void) {
	if (2 + 3 * 4 != 14) return 1;
	if ((2 + 3) * 4 != 20) return 1;
	if (10 - 2 * 3 != 4) return 1;
	if (10 / 2 + 3 != 8) return 1;
	if (10 % 3 + 1 != 2) return 1;

	/* shift vs addition */
	if ((1 << 3) + 1 != 9) return 1;

	/* comparison vs arithmetic */
	int r = 3 + 4 > 5;
	if (r != 1) return 1;

	/* bitwise vs comparison */
	r = (5 & 3) == 1;
	if (r != 1) return 1;

	return 0;
}
