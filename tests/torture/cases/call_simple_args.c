/* Adapted from gcc.dg/torture/call_simple_args.c -- tests calling convention */

int check(int a, int b, int c) {
	if (a != 10) return 1;
	if (b != 20) return 1;
	if (c != 30) return 1;
	return 0;
}

int main(void) {
	return check(10, 20, 30);
}
