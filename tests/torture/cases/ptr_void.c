/* Adapted from gcc.dg/torture/ptr_void.c -- tests pointers */
/* void pointer cast round-trip */

int main(void) {
	int x = 42;
	void *vp = &x;

	/* cast back to int* */
	int *ip = (int *)vp;
	if (*ip != 42) return 1;

	/* modify through casted pointer */
	*ip = 99;
	if (x != 99) return 1;

	/* char through void */
	char c = 'A';
	vp = &c;
	char *cp = (char *)vp;
	if (*cp != 65) return 1;

	return 0;
}
