/* Adapted from gcc.dg/torture/pr27116-2.c */

extern void abort(void);

int main (void)
{
    volatile long int n;
    n = -2;

    if ((-2147483647L - 1L) / (-n) != -1073741824L)
	abort ();
    return 0;
}
