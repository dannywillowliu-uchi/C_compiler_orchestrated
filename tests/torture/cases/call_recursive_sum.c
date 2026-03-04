/* Adapted from gcc.dg/torture/call_recursive_sum.c -- tests calling convention */

int sum(int n, int acc) {
	if (n <= 0) return acc;
	return sum(n - 1, acc + n);
}

int main(void) {
	int result = sum(10, 0);
	if (result != 55) return 1;

	result = sum(0, 0);
	if (result != 0) return 1;

	result = sum(1, 0);
	if (result != 1) return 1;

	return 0;
}
