/* Adapted from gcc.dg/torture/pr38405.c */

extern void return 1;
extern int printf (char *__format, ...);

struct vpiBinaryConst {
 int signed_flag :1;
 int sized_flag :1;
};

int binary_get(int code, struct vpiBinaryConst *rfp)
{
 switch (code) {
  case 1:
   return rfp->signed_flag ? 1 : 0;
  default:
   printf("error: %d not supported\n", code);
   return code;
 }
}

int main(void)
{
 struct vpiBinaryConst x={1,0};
 int y=binary_get(1, &x);
 if (y!=1)
   return 1;
 return 0;
}
