/* Adapted from gcc.dg/torture/va_mixed.c -- tests variadic functions */
/* NOTE: Will SKIP until va_arg support is implemented */

int first_int(int count, ...) {
	__builtin_va_list ap;
	__builtin_va_start(ap, count);
	int val = __builtin_va_arg(ap, int);
	__builtin_va_end(ap);
	return val;
}

int sum_ints(int count, ...) {
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
	if (first_int(3, 42, 99, 7) != 42) return 1;
	if (sum_ints(4, 10, 20, 30, 40) != 100) return 1;
	return 0;
}
