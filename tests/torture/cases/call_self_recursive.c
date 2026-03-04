/* Adapted from gcc.dg/torture/call_self_recursive.c -- tests calling convention */

int factorial(int n) {
	if (n <= 1) return 1;
	return n * factorial(n - 1);
}

int main(void) {
	if (factorial(1) != 1) return 1;
	if (factorial(2) != 2) return 1;
	if (factorial(3) != 6) return 1;
	if (factorial(5) != 120) return 1;
	if (factorial(7) != 5040) return 1;
	return 0;
}
