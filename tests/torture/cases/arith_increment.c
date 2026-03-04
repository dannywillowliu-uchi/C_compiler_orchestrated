/* Adapted from gcc.dg/torture/arith_increment.c -- tests arithmetic */

int main(void) {
	int a = 5;

	/* pre-increment */
	int r = ++a;
	if (r != 6) return 1;
	if (a != 6) return 1;

	/* post-increment */
	r = a++;
	if (r != 6) return 1;
	if (a != 7) return 1;

	/* pre-decrement */
	r = --a;
	if (r != 6) return 1;
	if (a != 6) return 1;

	/* post-decrement */
	r = a--;
	if (r != 6) return 1;
	if (a != 5) return 1;

	return 0;
}
