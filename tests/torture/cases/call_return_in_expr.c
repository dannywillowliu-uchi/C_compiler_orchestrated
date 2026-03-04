/* Adapted from gcc.dg/torture/call_return_in_expr.c -- tests calling convention */

int square(int x) {
	return x * x;
}

int add(int a, int b) {
	return a + b;
}

int main(void) {
	int r = square(3) + square(4);
	if (r != 25) return 1;

	r = add(square(2), square(3));
	if (r != 13) return 1;

	r = square(3) * 2 + 1;
	if (r != 19) return 1;

	return 0;
}
