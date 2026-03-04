/* Adapted from gcc.dg/torture/misc_comma.c -- tests miscellaneous */

int main(void) {
	/* comma operator: evaluates left, discards, returns right */
	int x = (1, 2, 3);
	if (x != 3) return 1;

	/* comma with side effects */
	int a = 0;
	int b = (a = 10, a + 5);
	if (a != 10) return 1;
	if (b != 15) return 1;

	/* comma in for loop */
	int sum = 0;
	int count = 0;
	for (int i = 0; i < 5; i = i + 1, count = count + 1) {
		sum = sum + i;
	}
	if (sum != 10) return 1;
	if (count != 5) return 1;

	return 0;
}
