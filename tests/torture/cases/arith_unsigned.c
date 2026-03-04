/* Adapted from gcc.dg/torture/arith_unsigned.c -- tests arithmetic */

int main(void) {
	unsigned int a = 0;
	unsigned int b = a - 1;  /* wraps to max unsigned */

	/* b should be UINT_MAX, which is at least 0xFFFFFFFF on 32-bit */
	if (b == 0) return 1;
	if (b + 1 != 0) return 1;  /* wrap around */

	unsigned int c = 10;
	unsigned int d = 3;
	if (c / d != 3) return 1;
	if (c % d != 1) return 1;

	/* unsigned comparison */
	unsigned int large = b;
	if (large < c) return 1;  /* max uint is larger than 10 */

	return 0;
}
