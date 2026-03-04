/* Adapted from gcc.dg/torture/misc_large_local.c -- tests miscellaneous */
/* Large local arrays test stack usage */

int main(void) {
	int arr[100];

	/* fill */
	for (int i = 0; i < 100; i = i + 1) {
		arr[i] = i * i;
	}

	/* verify */
	if (arr[0] != 0) return 1;
	if (arr[1] != 1) return 1;
	if (arr[10] != 100) return 1;
	if (arr[50] != 2500) return 1;
	if (arr[99] != 9801) return 1;

	/* sum first 10 */
	int sum = 0;
	for (int i = 0; i < 10; i = i + 1) {
		sum = sum + arr[i];
	}
	/* 0+1+4+9+16+25+36+49+64+81 = 285 */
	if (sum != 285) return 1;

	return 0;
}
