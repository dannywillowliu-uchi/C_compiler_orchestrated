/* Adapted from gcc.dg/torture/ptr_function.c -- tests pointers */
/* Function pointer basics */

int add(int a, int b) {
	return a + b;
}

int mul(int a, int b) {
	return a * b;
}

int apply(int (*fn)(int, int), int x, int y) {
	return fn(x, y);
}

int main(void) {
	int (*fp)(int, int);

	fp = add;
	if (fp(3, 4) != 7) return 1;

	fp = mul;
	if (fp(3, 4) != 12) return 1;

	/* through apply */
	if (apply(add, 10, 20) != 30) return 1;
	if (apply(mul, 10, 20) != 200) return 1;

	return 0;
}
