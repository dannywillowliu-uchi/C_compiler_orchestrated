/* Adapted from gcc.dg/torture/decl_global_local.c -- tests declarations */

int x = 100;

int get_global(void) {
	return x;
}

int main(void) {
	if (x != 100) return 1;
	if (get_global() != 100) return 1;

	int x = 42;  /* shadows global */
	if (x != 42) return 1;

	/* global is still 100, accessible via function */
	if (get_global() != 100) return 1;

	return 0;
}
