/* Adapted from gcc.dg/torture/flow_nested_loops.c -- tests control flow */

int main(void) {
	/* nested for loops */
	int sum = 0;
	for (int i = 0; i < 3; i = i + 1) {
		for (int j = 0; j < 4; j = j + 1) {
			sum = sum + 1;
		}
	}
	if (sum != 12) return 1;

	/* break in inner loop */
	int found_i = -1;
	int found_j = -1;
	for (int i = 0; i < 5; i = i + 1) {
		for (int j = 0; j < 5; j = j + 1) {
			if (i * 5 + j == 13) {
				found_i = i;
				found_j = j;
				break;
			}
		}
		if (found_i >= 0) break;
	}
	if (found_i != 2) return 1;
	if (found_j != 3) return 1;

	/* continue in inner loop */
	int count = 0;
	for (int i = 0; i < 5; i = i + 1) {
		for (int j = 0; j < 5; j = j + 1) {
			if (j == 2) continue;
			count = count + 1;
		}
	}
	if (count != 20) return 1;  /* 5 * 4 = 20 */

	return 0;
}
