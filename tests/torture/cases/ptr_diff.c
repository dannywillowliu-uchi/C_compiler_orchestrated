/* Adapted from gcc.dg/torture/ptr_diff.c -- tests pointers */

int main(void) {
	int arr[10];
	int *p0 = &arr[0];
	int *p5 = &arr[5];
	int *p9 = &arr[9];

	if (p5 - p0 != 5) return 1;
	if (p9 - p0 != 9) return 1;
	if (p9 - p5 != 4) return 1;

	/* negative difference */
	if (p0 - p5 != -5) return 1;

	/* zero difference */
	if (p0 - p0 != 0) return 1;

	/* char array - each element is 1 byte */
	char buf[10];
	char *c0 = &buf[0];
	char *c7 = &buf[7];
	if (c7 - c0 != 7) return 1;

	return 0;
}
