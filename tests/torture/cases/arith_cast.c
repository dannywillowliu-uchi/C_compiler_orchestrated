/* Adapted from gcc.dg/torture/arith_cast.c -- tests arithmetic */

int main(void) {
	/* int to char truncation */
	int big = 321;
	char c = (char)big;
	/* 321 % 256 = 65 = 'A' */
	if (c != 65) return 1;

	/* char to int (sign extension or zero extension) */
	char ch = 'Z';  /* 90 */
	int i = (int)ch;
	if (i != 90) return 1;

	/* unsigned to signed */
	unsigned int u = 42;
	int s = (int)u;
	if (s != 42) return 1;

	/* signed to unsigned */
	int neg = -1;
	unsigned int un = (unsigned int)neg;
	if (un == 0) return 1;  /* should be non-zero (max uint) */
	if (un + 1 != 0) return 1;  /* max uint + 1 wraps to 0 */

	return 0;
}
