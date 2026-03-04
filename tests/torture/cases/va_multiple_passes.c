/* Adapted from gcc.dg/torture/va_multiple_passes.c -- tests variadic functions */
/* NOTE: Will SKIP until va_arg + va_copy support is implemented */

int sum_twice(int count, ...) {
	__builtin_va_list ap, ap2;
	__builtin_va_start(ap, count);
	__builtin_va_copy(ap2, ap);

	int sum1 = 0;
	for (int i = 0; i < count; i = i + 1) {
		sum1 = sum1 + __builtin_va_arg(ap, int);
	}

	int sum2 = 0;
	for (int i = 0; i < count; i = i + 1) {
		sum2 = sum2 + __builtin_va_arg(ap2, int);
	}

	__builtin_va_end(ap);
	__builtin_va_end(ap2);

	if (sum1 != sum2) return -1;
	return sum1;
}

int main(void) {
	int r = sum_twice(3, 10, 20, 30);
	if (r != 60) return 1;
	return 0;
}
