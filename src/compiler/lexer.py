"""Lexer for the C compiler. Tokenizes C source code into a list of Tokens."""

from compiler.tokens import KEYWORDS, IntegerSuffix, Token, TokenType

_SIMPLE_ESCAPES = {
	"n": "\n",
	"t": "\t",
	"\\": "\\",
	"'": "'",
	'"': '"',
	"a": "\a",
	"b": "\b",
	"f": "\f",
	"r": "\r",
	"v": "\v",
	"0": "\0",
}

_OCTAL_DIGITS = frozenset("01234567")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def interpret_c_escapes(raw: str) -> str:
	"""Interpret C escape sequences in a raw string (quotes already stripped).

	Handles: \\n \\t \\0 \\\\ \\' \\" \\a \\b \\f \\r \\v \\xHH \\NNN (octal).
	"""
	result: list[str] = []
	i = 0
	length = len(raw)
	while i < length:
		if raw[i] != "\\" or i + 1 >= length:
			result.append(raw[i])
			i += 1
			continue
		nxt = raw[i + 1]
		# Simple single-char escapes
		if nxt in _SIMPLE_ESCAPES:
			result.append(_SIMPLE_ESCAPES[nxt])
			i += 2
			# \0 could be start of octal sequence like \012
			if nxt == "0":
				octal = ""
				while len(octal) < 2 and i < length and raw[i] in _OCTAL_DIGITS:
					octal += raw[i]
					i += 1
				if octal:
					result[-1] = chr(int("0" + octal, 8))
			continue
		# Octal escape: \1-7 followed by up to 2 more octal digits
		if nxt in _OCTAL_DIGITS:
			i += 2  # skip backslash and first digit
			octal = nxt
			while len(octal) < 3 and i < length and raw[i] in _OCTAL_DIGITS:
				octal += raw[i]
				i += 1
			result.append(chr(int(octal, 8)))
			continue
		# Hex escape: \xHH (reads all hex digits, truncates to byte)
		if nxt == "x":
			i += 2
			hex_str = ""
			while i < length and raw[i] in _HEX_DIGITS:
				hex_str += raw[i]
				i += 1
			if hex_str:
				result.append(chr(int(hex_str, 16) & 0xFF))
			else:
				# \x with no digits - pass through
				result.append("\\x")
			continue
		# Unknown escape - pass through
		result.append("\\")
		result.append(nxt)
		i += 2
	return "".join(result)


class LexerError(Exception):
	"""Error raised when the lexer encounters invalid input."""

	def __init__(self, message: str, line: int, column: int) -> None:
		self.line = line
		self.column = column
		super().__init__(f"{message} at {line}:{column}")


class Lexer:
	"""Tokenizes C source code into a sequence of tokens."""

	def __init__(self, source: str) -> None:
		self.source = source
		self.pos = 0
		self.line = 1
		self.column = 1
		self.tokens: list[Token] = []

	def _peek(self, offset: int = 0) -> str:
		idx = self.pos + offset
		if idx >= len(self.source):
			return "\0"
		return self.source[idx]

	def _advance(self) -> str:
		ch = self.source[self.pos]
		self.pos += 1
		if ch == "\n":
			self.line += 1
			self.column = 1
		else:
			self.column += 1
		return ch

	def _at_end(self) -> bool:
		return self.pos >= len(self.source)

	def _match(self, expected: str) -> bool:
		if self._at_end() or self.source[self.pos] != expected:
			return False
		self._advance()
		return True

	def _make_token(
		self,
		token_type: TokenType,
		value: str,
		line: int,
		column: int,
		suffix: IntegerSuffix = IntegerSuffix.NONE,
	) -> Token:
		return Token(type=token_type, value=value, line=line, column=column, suffix=suffix)

	def _skip_whitespace(self) -> None:
		while not self._at_end() and self._peek() in " \t\n\r\f\v":
			self._advance()

	def _skip_line_comment(self) -> None:
		while not self._at_end() and self._peek() != "\n":
			self._advance()

	def _skip_block_comment(self, start_line: int, start_col: int) -> None:
		while not self._at_end():
			if self._peek() == "*" and self._peek(1) == "/":
				self._advance()  # *
				self._advance()  # /
				return
			self._advance()
		raise LexerError("Unterminated block comment", start_line, start_col)

	def _read_string(self) -> Token:
		start_line = self.line
		start_col = self.column
		self._advance()  # opening "
		value = '"'
		while not self._at_end():
			ch = self._peek()
			if ch == "\n":
				raise LexerError("Unterminated string literal", start_line, start_col)
			if ch == "\\":
				value += self._advance()  # backslash
				if self._at_end():
					raise LexerError("Unterminated string literal", start_line, start_col)
				value += self._advance()  # escaped char
				continue
			if ch == '"':
				value += self._advance()
				return self._make_token(TokenType.STRING_LITERAL, value, start_line, start_col)
			value += self._advance()
		raise LexerError("Unterminated string literal", start_line, start_col)

	def _read_char(self) -> Token:
		start_line = self.line
		start_col = self.column
		self._advance()  # opening '
		value = "'"
		while not self._at_end():
			ch = self._peek()
			if ch == "\n":
				raise LexerError("Unterminated character literal", start_line, start_col)
			if ch == "\\":
				value += self._advance()
				if self._at_end():
					raise LexerError("Unterminated character literal", start_line, start_col)
				value += self._advance()
				continue
			if ch == "'":
				value += self._advance()
				return self._make_token(TokenType.CHAR_LITERAL, value, start_line, start_col)
			value += self._advance()
		raise LexerError("Unterminated character literal", start_line, start_col)

	def _read_number(self) -> Token:
		start_line = self.line
		start_col = self.column
		start_pos = self.pos
		ch = self._peek()

		if ch == "0" and self._peek(1) in ("x", "X"):
			# Hex literal (integer or float)
			self._advance()  # 0
			self._advance()  # x/X
			has_digits = self._is_hex_digit(self._peek())
			while not self._at_end() and self._is_hex_digit(self._peek()):
				self._advance()

			is_hex_float = False
			if not self._at_end() and self._peek() == ".":
				is_hex_float = True
				self._advance()  # .
				while not self._at_end() and self._is_hex_digit(self._peek()):
					self._advance()
					has_digits = True

			if not has_digits:
				raise LexerError("Invalid hex literal", start_line, start_col)

			if not self._at_end() and self._peek() in ("p", "P"):
				is_hex_float = True
				self._advance()  # p/P
				if not self._at_end() and self._peek() in ("+", "-"):
					self._advance()
				if self._at_end() or not self._peek().isdigit():
					raise LexerError("Invalid hex float literal exponent", start_line, start_col)
				while not self._at_end() and self._peek().isdigit():
					self._advance()
			elif is_hex_float:
				raise LexerError("Hex float literal requires exponent", start_line, start_col)

			if is_hex_float:
				if not self._at_end() and self._peek() in ("f", "F", "l", "L"):
					self._advance()
				return self._make_token(
					TokenType.FLOAT_LITERAL,
					self.source[start_pos:self.pos],
					start_line,
					start_col,
				)

			num_end = self.pos
			suffix = self._consume_integer_suffix()
			return self._make_token(
				TokenType.INTEGER_LITERAL,
				self.source[start_pos:num_end],
				start_line,
				start_col,
				suffix=suffix,
			)

		if ch == "0" and self._peek(1) not in (".", "e", "E") and self._is_octal_digit(self._peek(1)):
			# Octal literal
			self._advance()  # 0
			while not self._at_end() and self._is_octal_digit(self._peek()):
				self._advance()
			num_end = self.pos
			suffix = self._consume_integer_suffix()
			return self._make_token(
				TokenType.INTEGER_LITERAL,
				self.source[start_pos:num_end],
				start_line,
				start_col,
				suffix=suffix,
			)

		# Decimal integer or float
		while not self._at_end() and self._peek().isdigit():
			self._advance()

		is_float = False
		if not self._at_end() and self._peek() == "." and self._peek(1) != ".":
			is_float = True
			self._advance()  # .
			while not self._at_end() and self._peek().isdigit():
				self._advance()

		if not self._at_end() and self._peek() in ("e", "E"):
			is_float = True
			self._advance()
			if not self._at_end() and self._peek() in ("+", "-"):
				self._advance()
			if self._at_end() or not self._peek().isdigit():
				raise LexerError("Invalid float literal exponent", start_line, start_col)
			while not self._at_end() and self._peek().isdigit():
				self._advance()

		if is_float:
			if not self._at_end() and self._peek() in ("f", "F", "l", "L"):
				self._advance()
			return self._make_token(
				TokenType.FLOAT_LITERAL,
				self.source[start_pos:self.pos],
				start_line,
				start_col,
			)

		num_end = self.pos
		suffix = self._consume_integer_suffix()
		return self._make_token(
			TokenType.INTEGER_LITERAL,
			self.source[start_pos:num_end],
			start_line,
			start_col,
			suffix=suffix,
		)

	def _consume_integer_suffix(self) -> IntegerSuffix:
		"""Consume and return the integer literal suffix (u/U, l/L, ll/LL, ul/UL, ull/ULL)."""
		has_u = False
		long_count = 0

		if not self._at_end() and self._peek() in ("u", "U"):
			self._advance()
			has_u = True
		if not self._at_end() and self._peek() in ("l", "L"):
			self._advance()
			long_count = 1
			if not self._at_end() and self._peek() in ("l", "L"):
				self._advance()
				long_count = 2
		elif not has_u and not self._at_end() and self._peek() in ("l", "L"):
			self._advance()
			long_count = 1
			if not self._at_end() and self._peek() in ("l", "L"):
				self._advance()
				long_count = 2
		if not has_u and not self._at_end() and self._peek() in ("u", "U"):
			self._advance()
			has_u = True

		if has_u and long_count == 2:
			return IntegerSuffix.ULL
		if has_u and long_count == 1:
			return IntegerSuffix.UL
		if has_u:
			return IntegerSuffix.U
		if long_count == 2:
			return IntegerSuffix.LL
		if long_count == 1:
			return IntegerSuffix.L
		return IntegerSuffix.NONE

	def _read_identifier_or_keyword(self) -> Token:
		start_line = self.line
		start_col = self.column
		start_pos = self.pos
		while not self._at_end() and (self._peek().isalnum() or self._peek() == "_"):
			self._advance()
		text = self.source[start_pos:self.pos]
		token_type = KEYWORDS.get(text, TokenType.IDENTIFIER)
		return self._make_token(token_type, text, start_line, start_col)

	@staticmethod
	def _is_hex_digit(ch: str) -> bool:
		return ch in "0123456789abcdefABCDEF"

	@staticmethod
	def _is_octal_digit(ch: str) -> bool:
		return ch in "01234567"

	def _read_operator_or_punctuation(self) -> Token:
		start_line = self.line
		start_col = self.column
		ch = self._advance()

		match ch:
			case "(":
				return self._make_token(TokenType.LPAREN, "(", start_line, start_col)
			case ")":
				return self._make_token(TokenType.RPAREN, ")", start_line, start_col)
			case "[":
				return self._make_token(TokenType.LBRACKET, "[", start_line, start_col)
			case "]":
				return self._make_token(TokenType.RBRACKET, "]", start_line, start_col)
			case "{":
				return self._make_token(TokenType.LBRACE, "{", start_line, start_col)
			case "}":
				return self._make_token(TokenType.RBRACE, "}", start_line, start_col)
			case ";":
				return self._make_token(TokenType.SEMICOLON, ";", start_line, start_col)
			case ":":
				return self._make_token(TokenType.COLON, ":", start_line, start_col)
			case ",":
				return self._make_token(TokenType.COMMA, ",", start_line, start_col)
			case "~":
				return self._make_token(TokenType.TILDE, "~", start_line, start_col)
			case "?":
				return self._make_token(TokenType.QUESTION, "?", start_line, start_col)
			case "#":
				return self._make_token(TokenType.HASH, "#", start_line, start_col)
			case "+":
				if self._match("+"):
					return self._make_token(TokenType.INCREMENT, "++", start_line, start_col)
				if self._match("="):
					return self._make_token(TokenType.PLUS_ASSIGN, "+=", start_line, start_col)
				return self._make_token(TokenType.PLUS, "+", start_line, start_col)
			case "-":
				if self._match("-"):
					return self._make_token(TokenType.DECREMENT, "--", start_line, start_col)
				if self._match("="):
					return self._make_token(TokenType.MINUS_ASSIGN, "-=", start_line, start_col)
				if self._match(">"):
					return self._make_token(TokenType.ARROW, "->", start_line, start_col)
				return self._make_token(TokenType.MINUS, "-", start_line, start_col)
			case "*":
				if self._match("="):
					return self._make_token(TokenType.STAR_ASSIGN, "*=", start_line, start_col)
				return self._make_token(TokenType.STAR, "*", start_line, start_col)
			case "/":
				if self._match("="):
					return self._make_token(TokenType.SLASH_ASSIGN, "/=", start_line, start_col)
				return self._make_token(TokenType.SLASH, "/", start_line, start_col)
			case "%":
				if self._match("="):
					return self._make_token(TokenType.PERCENT_ASSIGN, "%=", start_line, start_col)
				return self._make_token(TokenType.PERCENT, "%", start_line, start_col)
			case "&":
				if self._match("&"):
					return self._make_token(TokenType.AND, "&&", start_line, start_col)
				if self._match("="):
					return self._make_token(TokenType.AMP_ASSIGN, "&=", start_line, start_col)
				return self._make_token(TokenType.AMPERSAND, "&", start_line, start_col)
			case "|":
				if self._match("|"):
					return self._make_token(TokenType.OR, "||", start_line, start_col)
				if self._match("="):
					return self._make_token(TokenType.PIPE_ASSIGN, "|=", start_line, start_col)
				return self._make_token(TokenType.PIPE, "|", start_line, start_col)
			case "^":
				if self._match("="):
					return self._make_token(TokenType.CARET_ASSIGN, "^=", start_line, start_col)
				return self._make_token(TokenType.CARET, "^", start_line, start_col)
			case "!":
				if self._match("="):
					return self._make_token(TokenType.NOT_EQUAL, "!=", start_line, start_col)
				return self._make_token(TokenType.BANG, "!", start_line, start_col)
			case "=":
				if self._match("="):
					return self._make_token(TokenType.EQUAL, "==", start_line, start_col)
				return self._make_token(TokenType.ASSIGN, "=", start_line, start_col)
			case "<":
				if self._match("<"):
					if self._match("="):
						return self._make_token(TokenType.LSHIFT_ASSIGN, "<<=", start_line, start_col)
					return self._make_token(TokenType.LSHIFT, "<<", start_line, start_col)
				if self._match("="):
					return self._make_token(TokenType.LESS_EQUAL, "<=", start_line, start_col)
				return self._make_token(TokenType.LESS, "<", start_line, start_col)
			case ">":
				if self._match(">"):
					if self._match("="):
						return self._make_token(TokenType.RSHIFT_ASSIGN, ">>=", start_line, start_col)
					return self._make_token(TokenType.RSHIFT, ">>", start_line, start_col)
				if self._match("="):
					return self._make_token(TokenType.GREATER_EQUAL, ">=", start_line, start_col)
				return self._make_token(TokenType.GREATER, ">", start_line, start_col)
			case ".":
				if self._peek() == "." and self._peek(1) == ".":
					self._advance()
					self._advance()
					return self._make_token(TokenType.ELLIPSIS, "...", start_line, start_col)
				return self._make_token(TokenType.DOT, ".", start_line, start_col)
			case _:
				raise LexerError(f"Unexpected character {ch!r}", start_line, start_col)

	def tokenize(self) -> list[Token]:
		"""Tokenize the entire source string and return a list of tokens."""
		self.pos = 0
		self.line = 1
		self.column = 1
		self.tokens = []

		while not self._at_end():
			self._skip_whitespace()
			if self._at_end():
				break

			ch = self._peek()

			# Comments
			if ch == "/" and self._peek(1) == "/":
				self._advance()
				self._advance()
				self._skip_line_comment()
				continue
			if ch == "/" and self._peek(1) == "*":
				comment_line = self.line
				comment_col = self.column
				self._advance()
				self._advance()
				self._skip_block_comment(comment_line, comment_col)
				continue

			# String literal
			if ch == '"':
				self.tokens.append(self._read_string())
				continue

			# Char literal
			if ch == "'":
				self.tokens.append(self._read_char())
				continue

			# Number literal
			if ch.isdigit():
				self.tokens.append(self._read_number())
				continue

			# Dot could start a float like .5
			if ch == "." and self._peek(1).isdigit():
				self.tokens.append(self._read_number_starting_with_dot())
				continue

			# Identifier or keyword
			if ch.isalpha() or ch == "_":
				self.tokens.append(self._read_identifier_or_keyword())
				continue

			# Operators and punctuation
			self.tokens.append(self._read_operator_or_punctuation())

		self.tokens = self._concatenate_string_literals(self.tokens)
		self.tokens.append(self._make_token(TokenType.EOF, "", self.line, self.column))
		return self.tokens

	@staticmethod
	def _concatenate_string_literals(tokens: list[Token]) -> list[Token]:
		"""Merge adjacent STRING_LITERAL tokens (C standard phase 6)."""
		if not tokens:
			return tokens
		result: list[Token] = []
		i = 0
		while i < len(tokens):
			if tokens[i].type != TokenType.STRING_LITERAL:
				result.append(tokens[i])
				i += 1
				continue
			# Collect run of consecutive string literals
			first = tokens[i]
			parts = [first.value[1:-1]]  # strip quotes
			i += 1
			while i < len(tokens) and tokens[i].type == TokenType.STRING_LITERAL:
				parts.append(tokens[i].value[1:-1])
				i += 1
			merged_value = '"' + "".join(parts) + '"'
			result.append(Token(
				type=TokenType.STRING_LITERAL,
				value=merged_value,
				line=first.line,
				column=first.column,
			))
		return result

	def _read_number_starting_with_dot(self) -> Token:
		"""Handle floats that start with a dot like .5, .123e10."""
		start_line = self.line
		start_col = self.column
		start_pos = self.pos
		self._advance()  # .
		while not self._at_end() and self._peek().isdigit():
			self._advance()
		if not self._at_end() and self._peek() in ("e", "E"):
			self._advance()
			if not self._at_end() and self._peek() in ("+", "-"):
				self._advance()
			if self._at_end() or not self._peek().isdigit():
				raise LexerError("Invalid float literal exponent", start_line, start_col)
			while not self._at_end() and self._peek().isdigit():
				self._advance()
		if not self._at_end() and self._peek() in ("f", "F", "l", "L"):
			self._advance()
		return self._make_token(
			TokenType.FLOAT_LITERAL,
			self.source[start_pos:self.pos],
			start_line,
			start_col,
		)
