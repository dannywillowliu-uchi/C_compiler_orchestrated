/* Adapted from gcc.dg/torture/va_pass_through.c -- tests variadic functions */
/* NOTE: Will SKIP until va_arg support is implemented */

int vsum(__builtin_va_list ap, int count) {
	int total = 0;
	for (int i = 0; i < count; i = i + 1) {
		total = total + __builtin_va_arg(ap, int);
	}
	return total;
}

int sum(int count, ...) {
	__builtin_va_list ap;
	__builtin_va_start(ap, count);
	int result = vsum(ap, count);
	__builtin_va_end(ap);
	return result;
}

int main(void) {
	if (sum(3, 10, 20, 30) != 60) return 1;
	if (sum(1, 100) != 100) return 1;
	return 0;
}
