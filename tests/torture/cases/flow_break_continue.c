/* Adapted from gcc.dg/torture/flow_break_continue.c -- tests control flow */

int main(void) {
	/* break exits innermost loop only */
	int outer_count = 0;
	for (int i = 0; i < 5; i = i + 1) {
		outer_count = outer_count + 1;
		for (int j = 0; j < 100; j = j + 1) {
			if (j == 2) break;
		}
	}
	if (outer_count != 5) return 1;

	/* continue skips rest of iteration */
	int sum = 0;
	for (int i = 0; i < 10; i = i + 1) {
		if (i % 2 == 0) continue;  /* skip even */
		sum = sum + i;
	}
	/* 1 + 3 + 5 + 7 + 9 = 25 */
	if (sum != 25) return 1;

	/* break in while */
	int w = 0;
	while (1) {
		w = w + 1;
		if (w == 7) break;
	}
	if (w != 7) return 1;

	/* continue in while */
	int c = 0;
	int iterations = 0;
	while (c < 10) {
		c = c + 1;
		if (c % 3 == 0) continue;
		iterations = iterations + 1;
	}
	/* c goes 1..10, skip 3,6,9 => 7 iterations */
	if (iterations != 7) return 1;

	return 0;
}
