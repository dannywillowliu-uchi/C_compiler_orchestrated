/* Adapted from gcc.dg/torture/call_recursive_fib.c -- tests calling convention */

int fib(int n) {
	if (n <= 0) return 0;
	if (n == 1) return 1;
	return fib(n - 1) + fib(n - 2);
}

int main(void) {
	if (fib(0) != 0) return 1;
	if (fib(1) != 1) return 1;
	if (fib(2) != 1) return 1;
	if (fib(3) != 2) return 1;
	if (fib(4) != 3) return 1;
	if (fib(5) != 5) return 1;
	if (fib(10) != 55) return 1;
	return 0;
}
