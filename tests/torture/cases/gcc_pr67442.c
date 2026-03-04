/* Adapted from gcc.dg/torture/pr67442.c */

short foo[100];

int main()
{
  short* bar = &foo[50];
  short i = 1;
  short j = 1;
  short value = bar[8 - i * 2 * j];
  return value;
}
