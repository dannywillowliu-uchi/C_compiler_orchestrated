/* Adapted from gcc.dg/torture/flow_ternary.c -- tests control flow */

int main(void) {
	/* basic ternary */
	int a = 5;
	int b = 10;
	int max = (a > b) ? a : b;
	if (max != 10) return 1;

	/* ternary as rvalue */
	int x = (1 > 0) ? 42 : 99;
	if (x != 42) return 1;

	/* nested ternary */
	int val = 2;
	int result = (val == 1) ? 10 : (val == 2) ? 20 : 30;
	if (result != 20) return 1;

	/* ternary in function argument */
	int arr[2];
	arr[0] = 100;
	arr[1] = 200;
	int flag = 1;
	int picked = arr[flag ? 1 : 0];
	if (picked != 200) return 1;

	/* ternary with side effects (only one branch evaluated) */
	int y = 0;
	int z = 0;
	int r = 1 ? (y = 5) : (z = 5);
	if (y != 5) return 1;
	if (z != 0) return 1;  /* z should NOT be modified */

	return 0;
}
