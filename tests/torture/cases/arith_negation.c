/* Adapted from gcc.dg/torture/arith_negation.c -- tests arithmetic */

int main(void) {
	int a = 42;
	if (-a != -42) return 1;

	int b = -10;
	if (-b != 10) return 1;

	int c = 0;
	if (-c != 0) return 1;

	/* double negation */
	int d = 7;
	if (-(-d) != 7) return 1;

	/* negation in expressions */
	int e = 5;
	int f = 3;
	if (-(e + f) != -8) return 1;
	if ((-e) + f != -2) return 1;

	return 0;
}
