/* Adapted from gcc.dg/torture/flow_switch_fallthrough.c -- tests control flow */

int count_remaining(int start) {
	int count = 0;
	switch (start) {
	case 1: count++;
	case 2: count++;
	case 3: count++;
	case 4: count++;
	case 5: count++;
	}
	return count;
}

int main(void) {
	if (count_remaining(1) != 5) return 1;
	if (count_remaining(3) != 3) return 1;
	if (count_remaining(5) != 1) return 1;
	if (count_remaining(6) != 0) return 1;  /* no case matches */
	return 0;
}
