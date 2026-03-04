/* Adapted from gcc.dg/torture/call_mixed_operations.c -- tests calling convention */

int double_val(int x) {
	return x * 2;
}

int main(void) {
	int a = 5;
	int b = 10;

	int r1 = double_val(a);
	/* a and b must survive the call */
	int sum = a + b + r1;
	if (sum != 25) return 1;

	int r2 = double_val(b);
	/* a, b, r1 must survive */
	int total = a + b + r1 + r2;
	if (total != 45) return 1;

	return 0;
}
