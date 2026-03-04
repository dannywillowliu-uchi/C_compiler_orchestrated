/* Adapted from gcc.dg/torture/misc_string_literal.c -- tests miscellaneous */

int main(void) {
	char *s = "Hello";

	if (s[0] != 'H') return 1;
	if (s[1] != 'e') return 1;
	if (s[2] != 'l') return 1;
	if (s[3] != 'l') return 1;
	if (s[4] != 'o') return 1;
	if (s[5] != 0) return 1;  /* null terminator */

	/* string length manually */
	int len = 0;
	while (s[len] != 0) {
		len = len + 1;
	}
	if (len != 5) return 1;

	/* empty string */
	char *empty = "";
	if (empty[0] != 0) return 1;

	return 0;
}
