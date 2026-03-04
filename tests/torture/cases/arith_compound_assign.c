/* Adapted from gcc.dg/torture/arith_compound_assign.c -- tests arithmetic */

int main(void) {
	int x = 10;

	x += 5;
	if (x != 15) return 1;

	x -= 3;
	if (x != 12) return 1;

	x *= 2;
	if (x != 24) return 1;

	x /= 4;
	if (x != 6) return 1;

	x %= 4;
	if (x != 2) return 1;

	x <<= 3;
	if (x != 16) return 1;

	x >>= 2;
	if (x != 4) return 1;

	x &= 0xFF;
	if (x != 4) return 1;

	x |= 0x10;
	if (x != 20) return 1;

	x ^= 0xFF;
	if (x != 235) return 1;

	return 0;
}
