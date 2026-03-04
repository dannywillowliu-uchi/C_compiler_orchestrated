/* Adapted from gcc.dg/torture/misc_typedef.c -- tests miscellaneous */

typedef int i32;
typedef unsigned int u32;
typedef int *intptr;

typedef struct {
	i32 x;
	i32 y;
} Point;

int main(void) {
	i32 a = 42;
	if (a != 42) return 1;

	u32 b = 100;
	if (b != 100) return 1;

	i32 val = 99;
	intptr p = &val;
	if (*p != 99) return 1;

	Point pt;
	pt.x = 10;
	pt.y = 20;
	if (pt.x + pt.y != 30) return 1;

	return 0;
}
