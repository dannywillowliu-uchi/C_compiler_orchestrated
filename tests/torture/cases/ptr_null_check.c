/* Adapted from gcc.dg/torture/ptr_null_check.c -- tests pointers */

int main(void) {
	int *p = 0;  /* null pointer */

	if (p != 0) return 1;
	if (p) return 1;  /* null is falsy */

	int x = 42;
	p = &x;

	if (p == 0) return 1;
	if (!p) return 1;  /* non-null is truthy */

	/* set back to null */
	p = 0;
	if (p != 0) return 1;

	return 0;
}
