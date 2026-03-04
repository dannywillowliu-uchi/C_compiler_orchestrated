/* Adapted from gcc.dg/torture/flow_switch_basic.c -- tests control flow */

int day_type(int day) {
	switch (day) {
	case 1: return 1;  /* Monday */
	case 2: return 1;
	case 3: return 1;
	case 4: return 1;
	case 5: return 1;
	case 6: return 2;  /* Saturday */
	case 7: return 2;  /* Sunday */
	default: return -1;
	}
}

int main(void) {
	if (day_type(1) != 1) return 1;
	if (day_type(5) != 1) return 1;
	if (day_type(6) != 2) return 1;
	if (day_type(7) != 2) return 1;
	if (day_type(0) != -1) return 1;
	if (day_type(8) != -1) return 1;
	return 0;
}
