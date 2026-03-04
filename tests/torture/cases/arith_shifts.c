/* Adapted from gcc.dg/torture/arith_shifts.c -- tests arithmetic */

int main(void) {
	if ((1 << 0) != 1) return 1;
	if ((1 << 1) != 2) return 1;
	if ((1 << 3) != 8) return 1;
	if ((1 << 10) != 1024) return 1;

	if ((16 >> 1) != 8) return 1;
	if ((16 >> 2) != 4) return 1;
	if ((16 >> 4) != 1) return 1;
	if ((255 >> 4) != 15) return 1;

	/* shift and mask */
	int x = 0xFF00;
	if ((x >> 8) != 0xFF) return 1;

	int y = 5;
	if ((y << 2) != 20) return 1;

	return 0;
}
