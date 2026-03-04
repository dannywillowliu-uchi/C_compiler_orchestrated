/* Adapted from gcc.dg/torture/ptr_arithmetic.c -- tests pointers */
/* p + 1 where p is int* should advance by sizeof(int) bytes */

int main(void) {
	int arr[5];
	arr[0] = 10;
	arr[1] = 20;
	arr[2] = 30;
	arr[3] = 40;
	arr[4] = 50;

	int *p = &arr[0];
	if (*p != 10) return 1;

	p = p + 1;
	if (*p != 20) return 1;

	p = p + 2;
	if (*p != 40) return 1;

	/* pointer subtraction from base */
	int *base = &arr[0];
	int *end = &arr[4];
	if (end - base != 4) return 1;

	return 0;
}
