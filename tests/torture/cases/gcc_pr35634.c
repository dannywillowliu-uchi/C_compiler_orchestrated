/* Adapted from gcc.dg/torture/pr35634.c */

void abort (void);
void exit (int);

void foo (int i)
{
    static int n;
    if (i < -128 || i > 127)
        return 1;
    if (++n > 1000)
        return 0;
}

int main ()
{
    signed char c;
    for (c = 0; ; c++) foo (c);
}
