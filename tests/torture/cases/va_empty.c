/* Adapted from gcc.dg/torture/va_empty.c -- tests variadic functions */
/* NOTE: Will SKIP until va_arg support is implemented */

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
	/* zero extra args */
	if (sum(0) != 0) return 1;
	/* one arg for sanity */
	if (sum(1, 42) != 42) return 1;
	return 0;
}
