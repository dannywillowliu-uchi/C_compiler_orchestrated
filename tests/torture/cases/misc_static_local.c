/* Adapted from gcc.dg/torture/misc_static_local.c -- tests miscellaneous */

int counter(void) {
	static int count = 0;
	count = count + 1;
	return count;
}

int main(void) {
	if (counter() != 1) return 1;
	if (counter() != 2) return 1;
	if (counter() != 3) return 1;
	if (counter() != 4) return 1;
	if (counter() != 5) return 1;
	return 0;
}
