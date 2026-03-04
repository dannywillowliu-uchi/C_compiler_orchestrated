/* Adapted from gcc.dg/torture/decl_mixed_types.c -- tests declarations */

int main(void) {
	int a = 10;
	char b = 'A';
	unsigned int c = 42;
	int d = -5;

	if (a != 10) return 1;
	if (b != 65) return 1;  /* 'A' == 65 */
	if (c != 42) return 1;
	if (d != -5) return 1;

	long e = 100000;
	short f = 32000;
	if (e != 100000) return 1;
	if (f != 32000) return 1;

	return 0;
}
