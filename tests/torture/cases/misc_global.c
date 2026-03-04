/* Adapted from gcc.dg/torture/misc_global.c -- tests miscellaneous */

int g_val = 42;
int g_zero;  /* should be zero-initialized */
int g_arr[3];

void set_global(int v) {
	g_val = v;
}

int main(void) {
	if (g_val != 42) return 1;
	if (g_zero != 0) return 1;

	set_global(99);
	if (g_val != 99) return 1;

	g_arr[0] = 10;
	g_arr[1] = 20;
	g_arr[2] = 30;
	int sum = g_arr[0] + g_arr[1] + g_arr[2];
	if (sum != 60) return 1;

	return 0;
}
