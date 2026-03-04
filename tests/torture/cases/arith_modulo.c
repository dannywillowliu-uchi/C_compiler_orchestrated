/* Adapted from gcc.dg/torture/arith_modulo.c -- tests arithmetic */

int main(void) {
	if (10 % 3 != 1) return 1;
	if (10 % 5 != 0) return 1;
	if (7 % 2 != 1) return 1;
	if (100 % 7 != 2) return 1;
	if (1 % 1 != 0) return 1;

	/* C99: (a/b)*b + a%b == a */
	int a = 17, b = 5;
	if ((a / b) * b + (a % b) != a) return 1;

	a = -17;
	b = 5;
	if ((a / b) * b + (a % b) != a) return 1;

	return 0;
}
