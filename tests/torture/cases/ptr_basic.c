/* Adapted from gcc.dg/torture/ptr_basic.c -- tests pointers */

int main(void) {
	int x = 42;
	int *p = &x;

	if (*p != 42) return 1;

	*p = 99;
	if (x != 99) return 1;

	int y = 10;
	p = &y;
	if (*p != 10) return 1;

	*p = *p + 5;
	if (y != 15) return 1;

	return 0;
}
