/* Adapted from gcc.dg/torture/va_simple.c -- tests variadic functions */
/* NOTE: Will SKIP until va_arg support is implemented */
/* Uses va_list/va_start/va_arg/va_end directly without headers */

int sum(int count, ...) {
	__builtin_va_list ap;
	__builtin_va_start(ap, count);
	int total = 0;
	for (int i = 0; i < count; i = i + 1) {
		total = total + __builtin_va_arg(ap, int);
	}
	__builtin_va_end(ap);
	return total;
}

int main(void) {
	if (sum(3, 10, 20, 30) != 60) return 1;
	if (sum(1, 42) != 42) return 1;
	if (sum(5, 1, 2, 3, 4, 5) != 15) return 1;
	return 0;
}
