/* Adapted from gcc.dg/torture/decl_for_init.c -- tests declarations */

int main(void) {
	int sum = 0;

	for (int i = 0; i < 5; i = i + 1) {
		sum = sum + i;
	}
	if (sum != 10) return 1;

	int product = 1;
	for (int j = 1; j <= 5; j = j + 1) {
		product = product * j;
	}
	if (product != 120) return 1;

	return 0;
}
