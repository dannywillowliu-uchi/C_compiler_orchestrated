/* Adapted from gcc.dg/torture/call_many_args.c -- tests calling convention */

int check(int a, int b, int c, int d, int e, int f, int g, int h) {
	if (a != 1) return 1;
	if (b != 2) return 1;
	if (c != 3) return 1;
	if (d != 4) return 1;
	if (e != 5) return 1;
	if (f != 6) return 1;
	if (g != 7) return 1;
	if (h != 8) return 1;
	return 0;
}

int main(void) {
	return check(1, 2, 3, 4, 5, 6, 7, 8);
}
