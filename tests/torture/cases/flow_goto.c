/* Adapted from gcc.dg/torture/flow_goto.c -- tests control flow */

int main(void) {
	int x = 0;

	/* forward goto */
	goto skip;
	x = 99;
skip:
	if (x != 0) return 1;

	/* backward goto (simple loop) */
	int count = 0;
loop:
	if (count >= 5) goto done;
	count = count + 1;
	goto loop;
done:
	if (count != 5) return 1;

	/* goto out of nested block */
	for (int i = 0; i < 10; i = i + 1) {
		if (i == 3) goto exit_loop;
	}
	return 1;  /* should not reach here */
exit_loop:

	return 0;
}
