/* Adapted from gcc.dg/torture/decl_multi_ptr.c -- tests declarations */
/* In `int *a, b;` -- a is int*, b is int (NOT int*) */

int main(void) {
	int val = 42;
	int *a, b;
	a = &val;
	b = 99;

	if (*a != 42) return 1;
	if (b != 99) return 1;

	/* Verify b is an int, not a pointer */
	b = *a + 1;
	if (b != 43) return 1;

	return 0;
}
