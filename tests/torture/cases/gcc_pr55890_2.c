/* Adapted from gcc.dg/torture/pr55890-2.c */
/* { dg-do compile } */

extern void *memcpy();
int main() { memcpy(); }

