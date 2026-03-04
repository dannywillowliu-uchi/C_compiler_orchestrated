/* Adapted from gcc.dg/torture/flow_short_circuit.c -- tests control flow */

int call_count;

int side_effect(int val) {
	call_count = call_count + 1;
	return val;
}

int main(void) {
	/* && short-circuit: if left is false, right is NOT evaluated */
	call_count = 0;
	if (0 && side_effect(1)) return 1;
	if (call_count != 0) return 1;  /* side_effect should NOT have been called */

	/* && non-short-circuit: if left is true, right IS evaluated */
	call_count = 0;
	int r = 1 && side_effect(1);
	if (call_count != 1) return 1;

	/* || short-circuit: if left is true, right is NOT evaluated */
	call_count = 0;
	r = 1 || side_effect(1);
	if (call_count != 0) return 1;

	/* || non-short-circuit: if left is false, right IS evaluated */
	call_count = 0;
	r = 0 || side_effect(1);
	if (call_count != 1) return 1;
	if (r != 1) return 1;

	return 0;
}
