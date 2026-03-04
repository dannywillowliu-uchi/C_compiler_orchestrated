/* Adapted from gcc.dg/torture/call_chain.c -- tests calling convention */

int c_func(int x) {
	return x + 3;
}

int b_func(int x) {
	return c_func(x) + 2;
}

int a_func(int x) {
	return b_func(x) + 1;
}

int main(void) {
	int result = a_func(10);
	/* 10 + 3 + 2 + 1 = 16 */
	if (result != 16) return 1;

	result = a_func(0);
	if (result != 6) return 1;

	return 0;
}
