/* Adapted from gcc.dg/torture/arith_logical.c -- tests arithmetic */

int main(void) {
	/* logical AND */
	if (!(1 && 1)) return 1;
	if (0 && 1) return 1;
	if (1 && 0) return 1;
	if (0 && 0) return 1;

	/* logical OR */
	if (!(1 || 0)) return 1;
	if (!(0 || 1)) return 1;
	if (!(1 || 1)) return 1;
	if (0 || 0) return 1;

	/* logical NOT */
	if (!!0) return 1;
	if (!1) return 1;
	if (!42) return 1;  /* non-zero is truthy */

	/* truthy non-zero values */
	if (!(5 && -1)) return 1;
	if (!(100 || 0)) return 1;

	return 0;
}
