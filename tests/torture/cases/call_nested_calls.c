/* Adapted from gcc.dg/torture/call_nested_calls.c -- tests calling convention */

int add(int a, int b) {
	return a + b;
}

int mul(int a, int b) {
	return a * b;
}

int combine(int x, int y) {
	return x + y;
}

int main(void) {
	int result = combine(add(1, 2), mul(3, 4));
	if (result != 15) return 1;

	result = combine(add(10, 20), mul(5, 6));
	if (result != 60) return 1;

	return 0;
}
