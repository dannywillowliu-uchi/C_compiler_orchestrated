/* Adapted from gcc.dg/torture/call_args_after_call.c -- tests calling convention */
/* Tests that parameters survive across a function call (register clobbering) */

int clobber(int a, int b, int c) {
	return a + b + c;
}

int test(int x, int y, int z) {
	int tmp = clobber(100, 200, 300);
	/* x, y, z must still be correct after the call */
	if (x != 10) return 1;
	if (y != 20) return 1;
	if (z != 30) return 1;
	if (tmp != 600) return 1;
	return 0;
}

int main(void) {
	return test(10, 20, 30);
}
