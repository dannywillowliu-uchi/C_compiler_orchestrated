/* Adapted from gcc.dg/torture/misc_enum.c -- tests miscellaneous */

enum Color {
	RED,     /* 0 */
	GREEN,   /* 1 */
	BLUE     /* 2 */
};

enum Status {
	ERR = -1,
	OK = 0,
	PENDING = 100
};

int main(void) {
	enum Color c = RED;
	if (c != 0) return 1;

	c = GREEN;
	if (c != 1) return 1;

	c = BLUE;
	if (c != 2) return 1;

	enum Status s = OK;
	if (s != 0) return 1;

	s = ERR;
	if (s != -1) return 1;

	s = PENDING;
	if (s != 100) return 1;

	return 0;
}
