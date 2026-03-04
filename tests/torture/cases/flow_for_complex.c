/* Adapted from gcc.dg/torture/flow_for_complex.c -- tests control flow */

int main(void) {
	/* multiple updates in for */
	int a = 0;
	int b = 10;
	for (int i = 0; i < 5; i = i + 1) {
		a = a + 1;
		b = b - 1;
	}
	if (a != 5) return 1;
	if (b != 5) return 1;

	/* empty body */
	int sum = 0;
	for (sum = 0; sum < 10; sum = sum + 1) {
	}
	if (sum != 10) return 1;

	/* while-style for */
	int n = 100;
	int steps = 0;
	for (; n > 1; ) {
		if (n % 2 == 0) {
			n = n / 2;
		} else {
			n = 3 * n + 1;
		}
		steps = steps + 1;
		if (steps > 200) return 1;  /* safety */
	}
	if (n != 1) return 1;

	return 0;
}
