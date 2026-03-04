/* Adapted from gcc.dg/torture/arith_sizeof.c -- tests arithmetic */

int main(void) {
	/* sizeof(char) is always 1 by definition */
	if (sizeof(char) != 1) return 1;

	/* sizeof(int) is typically 4 but at least 2 */
	if (sizeof(int) < 2) return 1;

	/* pointer size should be consistent */
	int x;
	int *p = &x;
	if (sizeof(p) < 4) return 1;  /* at least 32-bit pointers */

	/* sizeof array = element_size * count */
	int arr[10];
	if (sizeof(arr) != sizeof(int) * 10) return 1;

	/* sizeof(char) array */
	char buf[20];
	if (sizeof(buf) != 20) return 1;

	return 0;
}
