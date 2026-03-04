/* Adapted from gcc.dg/torture/call_side_effects.c -- tests calling convention */

int counter;

int next(void) {
	counter = counter + 1;
	return counter;
}

int main(void) {
	counter = 0;

	int a = next();
	int b = next();
	int c = next();

	if (a != 1) return 1;
	if (b != 2) return 1;
	if (c != 3) return 1;

	return 0;
}
