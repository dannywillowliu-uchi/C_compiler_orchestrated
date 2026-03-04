/* Adapted from gcc.dg/torture/arith_comparison.c -- tests arithmetic */

int main(void) {
	if (!(5 > 3)) return 1;
	if (!(3 < 5)) return 1;
	if (!(5 >= 5)) return 1;
	if (!(5 >= 3)) return 1;
	if (!(5 <= 5)) return 1;
	if (!(3 <= 5)) return 1;
	if (!(5 == 5)) return 1;
	if (!(5 != 3)) return 1;

	/* false cases */
	if (3 > 5) return 1;
	if (5 < 3) return 1;
	if (3 >= 5) return 1;
	if (5 <= 3) return 1;
	if (3 == 5) return 1;
	if (5 != 5) return 1;

	/* negative numbers */
	if (!(-1 < 0)) return 1;
	if (!(-5 < -3)) return 1;
	if (!(-3 > -5)) return 1;

	return 0;
}
