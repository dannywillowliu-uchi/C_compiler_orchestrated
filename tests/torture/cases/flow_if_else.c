/* Adapted from gcc.dg/torture/flow_if_else.c -- tests control flow */

int classify(int x) {
	if (x > 0) {
		return 1;
	} else if (x < 0) {
		return -1;
	} else {
		return 0;
	}
}

int main(void) {
	if (classify(10) != 1) return 1;
	if (classify(-5) != -1) return 1;
	if (classify(0) != 0) return 1;

	/* nested if */
	int a = 10;
	int b = 20;
	int max;
	if (a > b) {
		max = a;
	} else {
		max = b;
	}
	if (max != 20) return 1;

	/* chain of else-if */
	int val = 3;
	int result;
	if (val == 1) result = 10;
	else if (val == 2) result = 20;
	else if (val == 3) result = 30;
	else result = -1;
	if (result != 30) return 1;

	return 0;
}
