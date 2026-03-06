/* Adapted from gcc.dg/torture/pr53790.c */

typedef struct s {
    int value;
} s_t;

union u {
    int value;
};
union u extern_var;

static inline int
readval(s_t const *var)
{
  return var->value;
}

int main()
{
  return readval((s_t *)&extern_var);
}
