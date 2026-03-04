/* Adapted from gcc.dg/torture/ptr_double.c -- tests pointers */
/* Pointer to pointer (int **) */

void set_value(int **pp, int *target) {
	*pp = target;
}

int main(void) {
	int a = 10;
	int b = 20;
	int *p = &a;
	int **pp = &p;

	if (**pp != 10) return 1;

	/* change what p points to via pp */
	*pp = &b;
	if (*p != 20) return 1;
	if (**pp != 20) return 1;

	/* modify through double pointer */
	**pp = 99;
	if (b != 99) return 1;

	/* use function */
	int c = 42;
	set_value(pp, &c);
	if (*p != 42) return 1;

	return 0;
}
