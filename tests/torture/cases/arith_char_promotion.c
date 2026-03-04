/* Adapted from gcc.dg/torture/arith_char_promotion.c -- tests arithmetic */
/* char promotes to int in expressions */

int main(void) {
	char a = 100;
	char b = 50;

	/* result of a + b is int, not char */
	int r = a + b;
	if (r != 150) return 1;

	char c = 10;
	char d = 20;
	int product = c * d;
	if (product != 200) return 1;

	/* char comparison promotes to int */
	char x = 'A';  /* 65 */
	char y = 'Z';  /* 90 */
	if (y - x != 25) return 1;

	return 0;
}
