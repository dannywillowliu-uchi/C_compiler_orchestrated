/* Adapted from gcc.dg/torture/arith_overflow.c -- tests arithmetic */
/* Tests unsigned overflow wrapping (well-defined in C) */

int main(void) {
	unsigned int max = (unsigned int)-1;

	/* max + 1 wraps to 0 */
	if (max + 1 != 0) return 1;

	/* 0 - 1 wraps to max */
	unsigned int zero = 0;
	if (zero - 1 != max) return 1;

	/* multiplication wrapping */
	unsigned int a = max;
	unsigned int b = a * 2;
	/* max * 2 = 2 * (2^32-1) = 2^33 - 2, mod 2^32 = 2^32 - 2 = max - 1 */
	if (b != max - 1) return 1;

	return 0;
}
