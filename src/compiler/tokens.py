"""Token definitions for the C compiler lexer."""

from dataclasses import dataclass
from enum import Enum, auto


class IntegerSuffix(Enum):
	"""Integer literal suffix indicating type."""

	NONE = ""
	U = "u"
	L = "l"
	UL = "ul"
	LL = "ll"
	ULL = "ull"


class TokenType(Enum):
	"""All token types for C89."""

	# Literals
	INTEGER_LITERAL = auto()
	FLOAT_LITERAL = auto()
	CHAR_LITERAL = auto()
	STRING_LITERAL = auto()

	# Identifier
	IDENTIFIER = auto()

	# C89 Keywords
	AUTO = auto()
	BOOL = auto()
	BREAK = auto()
	CASE = auto()
	CHAR = auto()
	CONST = auto()
	CONTINUE = auto()
	DEFAULT = auto()
	DO = auto()
	DOUBLE = auto()
	ELSE = auto()
	ENUM = auto()
	EXTERN = auto()
	FLOAT = auto()
	FOR = auto()
	GOTO = auto()
	IF = auto()
	INT = auto()
	LONG = auto()
	REGISTER = auto()
	RETURN = auto()
	SHORT = auto()
	SIGNED = auto()
	SIZEOF = auto()
	STATIC = auto()
	STRUCT = auto()
	SWITCH = auto()
	TYPEDEF = auto()
	UNION = auto()
	UNSIGNED = auto()
	VOID = auto()
	VOLATILE = auto()
	WHILE = auto()
	INLINE = auto()        # inline (C99)
	STATIC_ASSERT = auto()  # _Static_assert (C11)
	NULLPTR = auto()       # nullptr (C23)

	# Operators
	PLUS = auto()          # +
	MINUS = auto()         # -
	STAR = auto()          # *
	SLASH = auto()         # /
	PERCENT = auto()       # %
	AMPERSAND = auto()     # &
	PIPE = auto()          # |
	CARET = auto()         # ^
	TILDE = auto()         # ~
	BANG = auto()           # !
	ASSIGN = auto()        # =
	LESS = auto()          # <
	GREATER = auto()       # >
	DOT = auto()           # .
	QUESTION = auto()      # ?

	# Compound operators
	PLUS_ASSIGN = auto()   # +=
	MINUS_ASSIGN = auto()  # -=
	STAR_ASSIGN = auto()   # *=
	SLASH_ASSIGN = auto()  # /=
	PERCENT_ASSIGN = auto()  # %=
	AMP_ASSIGN = auto()    # &=
	PIPE_ASSIGN = auto()   # |=
	CARET_ASSIGN = auto()  # ^=
	LSHIFT_ASSIGN = auto()  # <<=
	RSHIFT_ASSIGN = auto()  # >>=
	EQUAL = auto()         # ==
	NOT_EQUAL = auto()     # !=
	LESS_EQUAL = auto()    # <=
	GREATER_EQUAL = auto()  # >=
	AND = auto()           # &&
	OR = auto()            # ||
	LSHIFT = auto()        # <<
	RSHIFT = auto()        # >>
	INCREMENT = auto()     # ++
	DECREMENT = auto()     # --
	ARROW = auto()         # ->
	ELLIPSIS = auto()      # ...

	# Punctuation
	LPAREN = auto()        # (
	RPAREN = auto()        # )
	LBRACKET = auto()      # [
	RBRACKET = auto()      # ]
	LBRACE = auto()        # {
	RBRACE = auto()        # }
	SEMICOLON = auto()     # ;
	COLON = auto()         # :
	COMMA = auto()         # ,
	HASH = auto()          # #

	# Special
	EOF = auto()


# Map keyword strings to their token types
KEYWORDS: dict[str, TokenType] = {
	"_Bool": TokenType.BOOL,
	"bool": TokenType.BOOL,
	"auto": TokenType.AUTO,
	"break": TokenType.BREAK,
	"case": TokenType.CASE,
	"char": TokenType.CHAR,
	"const": TokenType.CONST,
	"continue": TokenType.CONTINUE,
	"default": TokenType.DEFAULT,
	"do": TokenType.DO,
	"double": TokenType.DOUBLE,
	"else": TokenType.ELSE,
	"enum": TokenType.ENUM,
	"extern": TokenType.EXTERN,
	"float": TokenType.FLOAT,
	"for": TokenType.FOR,
	"goto": TokenType.GOTO,
	"if": TokenType.IF,
	"int": TokenType.INT,
	"long": TokenType.LONG,
	"register": TokenType.REGISTER,
	"return": TokenType.RETURN,
	"short": TokenType.SHORT,
	"signed": TokenType.SIGNED,
	"sizeof": TokenType.SIZEOF,
	"static": TokenType.STATIC,
	"struct": TokenType.STRUCT,
	"switch": TokenType.SWITCH,
	"typedef": TokenType.TYPEDEF,
	"union": TokenType.UNION,
	"unsigned": TokenType.UNSIGNED,
	"void": TokenType.VOID,
	"volatile": TokenType.VOLATILE,
	"while": TokenType.WHILE,
	"inline": TokenType.INLINE,
	"__inline": TokenType.INLINE,
	"__inline__": TokenType.INLINE,
	"_Static_assert": TokenType.STATIC_ASSERT,
	"nullptr": TokenType.NULLPTR,
}


@dataclass(frozen=True)
class Token:
	"""A single token produced by the lexer."""

	type: TokenType
	value: str
	line: int
	column: int
	suffix: IntegerSuffix = IntegerSuffix.NONE

	def __repr__(self) -> str:
		return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"
