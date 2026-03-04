/* Adapted from gcc.dg/torture/call_seven_args.c -- tests calling convention */
/* 7 args: on x86-64, 6 go in registers, 7th goes on stack */

int check7(int a, int b, int c, int d, int e, int f, int g) {
	if (a != 11) return 1;
	if (b != 22) return 1;
	if (c != 33) return 1;
	if (d != 44) return 1;
	if (e != 55) return 1;
	if (f != 66) return 1;
	if (g != 77) return 1;
	return a + b + c + d + e + f + g;
}

int main(void) {
	int r = check7(11, 22, 33, 44, 55, 66, 77);
	if (r != 308) return 1;
	return 0;
}
