/* Adapted from gcc.dg/torture/flow_do_while.c -- tests control flow */

int main(void) {
	/* basic do-while */
	int x = 0;
	do {
		x = x + 1;
	} while (x < 5);
	if (x != 5) return 1;

	/* do-while executes at least once */
	int ran = 0;
	do {
		ran = 1;
	} while (0);
	if (ran != 1) return 1;

	/* do-while with break */
	int count = 0;
	do {
		count = count + 1;
		if (count == 3) break;
	} while (count < 100);
	if (count != 3) return 1;

	return 0;
}
