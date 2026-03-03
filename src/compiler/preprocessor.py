"""C preprocessor implementation.

Handles #define, #undef, #ifdef/#ifndef/#if/#elif/#else/#endif,
#include, __LINE__/__FILE__ predefined macros, macro expansion
with argument substitution, and line/comment stripping.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


class PreprocessorError(Exception):
	"""Error raised during preprocessing."""

	def __init__(self, message: str, filename: str = "<stdin>", line: int = 0) -> None:
		self.filename = filename
		self.line = line
		super().__init__(f"{filename}:{line}: {message}")


@dataclass
class Macro:
	"""Represents a preprocessor macro definition."""

	name: str
	body: str
	params: list[str] | None = None  # None = object-like, list = function-like
	is_variadic: bool = False


@dataclass
class _IfState:
	"""Tracks conditional compilation state."""

	active: bool  # Whether current branch is active
	seen_true: bool  # Whether any branch in this #if group was true
	is_else: bool = False  # Whether we've seen #else


class Preprocessor:
	"""C preprocessor that transforms source text for the lexer."""

	def __init__(
		self,
		include_paths: list[str] | None = None,
		predefined_macros: dict[str, str] | None = None,
	) -> None:
		self.include_paths: list[str] = include_paths or []
		self.macros: dict[str, Macro] = {}
		self._if_stack: list[_IfState] = []
		self._included_files: set[str] = set()

		if predefined_macros:
			for name, body in predefined_macros.items():
				self.macros[name] = Macro(name=name, body=body)

	def process(self, source: str, filename: str = "<stdin>") -> str:
		"""Process source text and return preprocessed output."""
		source = self._strip_comments(source)
		source = self._join_continuation_lines(source)
		lines = source.split("\n")
		output_lines: list[str] = []

		for line_num, line in enumerate(lines, start=1):
			stripped = line.strip()

			if stripped.startswith("#"):
				self._handle_directive(stripped, filename, line_num, output_lines)
			elif self._is_active():
				expanded = self._expand_macros(line, filename, line_num)
				output_lines.append(expanded)
			else:
				output_lines.append("")

		if self._if_stack:
			raise PreprocessorError("Unterminated conditional directive", filename, len(lines))

		return "\n".join(output_lines)

	def _is_active(self) -> bool:
		"""Check if current code section is active (not disabled by conditionals)."""
		return all(state.active for state in self._if_stack)

	def _strip_comments(self, source: str) -> str:
		"""Remove C-style comments from source, preserving line structure."""
		result: list[str] = []
		i = 0
		in_string = False
		in_char = False
		length = len(source)

		while i < length:
			ch = source[i]

			if in_string:
				result.append(ch)
				if ch == "\\" and i + 1 < length:
					i += 1
					result.append(source[i])
				elif ch == '"':
					in_string = False
				i += 1
				continue

			if in_char:
				result.append(ch)
				if ch == "\\" and i + 1 < length:
					i += 1
					result.append(source[i])
				elif ch == "'":
					in_char = False
				i += 1
				continue

			if ch == '"':
				in_string = True
				result.append(ch)
				i += 1
				continue

			if ch == "'":
				in_char = True
				result.append(ch)
				i += 1
				continue

			if ch == "/" and i + 1 < length:
				next_ch = source[i + 1]
				if next_ch == "/":
					# Line comment: skip to end of line
					while i < length and source[i] != "\n":
						i += 1
					continue
				elif next_ch == "*":
					# Block comment: replace with space, preserve newlines
					i += 2
					result.append(" ")
					while i < length:
						if source[i] == "*" and i + 1 < length and source[i + 1] == "/":
							i += 2
							break
						if source[i] == "\n":
							result.append("\n")
						i += 1
					continue

			result.append(ch)
			i += 1

		return "".join(result)

	def _join_continuation_lines(self, source: str) -> str:
		"""Join lines ending with backslash."""
		lines = source.split("\n")
		result: list[str] = []
		current = ""

		for line in lines:
			if line.endswith("\\"):
				current += line[:-1]
			else:
				current += line
				result.append(current)
				current = ""

		if current:
			result.append(current)

		return "\n".join(result)

	def _handle_directive(
		self,
		line: str,
		filename: str,
		line_num: int,
		output_lines: list[str],
	) -> None:
		"""Parse and execute a preprocessor directive."""
		# Strip the '#' and get the directive name
		rest = line[1:].strip()

		if not rest:
			output_lines.append("")
			return

		parts = rest.split(None, 1)
		directive = parts[0]
		args = parts[1] if len(parts) > 1 else ""

		# Conditional directives are always processed (they control active state)
		if directive == "ifdef":
			self._handle_ifdef(args, filename, line_num)
			output_lines.append("")
		elif directive == "ifndef":
			self._handle_ifndef(args, filename, line_num)
			output_lines.append("")
		elif directive == "if":
			self._handle_if(args, filename, line_num)
			output_lines.append("")
		elif directive == "elif":
			self._handle_elif(args, filename, line_num)
			output_lines.append("")
		elif directive == "else":
			self._handle_else(filename, line_num)
			output_lines.append("")
		elif directive == "endif":
			self._handle_endif(filename, line_num)
			output_lines.append("")
		elif self._is_active():
			if directive == "define":
				self._handle_define(args, filename, line_num)
				output_lines.append("")
			elif directive == "undef":
				self._handle_undef(args, filename, line_num)
				output_lines.append("")
			elif directive == "include":
				included = self._handle_include(args, filename, line_num)
				output_lines.append(included)
			elif directive == "error":
				raise PreprocessorError(f"#error {args}", filename, line_num)
			elif directive == "warning" or directive == "pragma" or directive == "line":
				output_lines.append("")
			else:
				raise PreprocessorError(f"Unknown directive #{directive}", filename, line_num)
		else:
			output_lines.append("")

	def _handle_define(self, args: str, filename: str, line_num: int) -> None:
		"""Handle #define directive."""
		if not args:
			raise PreprocessorError("Expected macro name after #define", filename, line_num)

		# Match: NAME or NAME(params) body
		# Function-like macros require '(' immediately after name (no space)
		match = re.match(r"([A-Za-z_]\w*)(\(.*?\))?\s*(.*)", args, re.DOTALL)
		if not match:
			raise PreprocessorError("Invalid #define syntax", filename, line_num)

		name = match.group(1)
		param_list = match.group(2)
		body = match.group(3).strip()

		params: list[str] | None = None
		is_variadic = False

		if param_list is not None:
			# Function-like macro
			inner = param_list[1:-1].strip()
			if inner:
				params = [p.strip() for p in inner.split(",")]
				if params and params[-1] == "...":
					is_variadic = True
					params.pop()
			else:
				params = []

		self.macros[name] = Macro(
			name=name,
			body=body,
			params=params,
			is_variadic=is_variadic,
		)

	def _handle_undef(self, args: str, filename: str, line_num: int) -> None:
		"""Handle #undef directive."""
		name = args.strip()
		if not name:
			raise PreprocessorError("Expected macro name after #undef", filename, line_num)
		self.macros.pop(name, None)

	def _handle_ifdef(self, args: str, filename: str, line_num: int) -> None:
		"""Handle #ifdef directive."""
		name = args.strip()
		if not name:
			raise PreprocessorError("Expected macro name after #ifdef", filename, line_num)

		parent_active = self._is_active()
		is_defined = name in self.macros
		active = parent_active and is_defined
		self._if_stack.append(_IfState(active=active, seen_true=active))

	def _handle_ifndef(self, args: str, filename: str, line_num: int) -> None:
		"""Handle #ifndef directive."""
		name = args.strip()
		if not name:
			raise PreprocessorError("Expected macro name after #ifndef", filename, line_num)

		parent_active = self._is_active()
		is_defined = name not in self.macros
		active = parent_active and is_defined
		self._if_stack.append(_IfState(active=active, seen_true=active))

	def _handle_if(self, args: str, filename: str, line_num: int) -> None:
		"""Handle #if directive."""
		parent_active = self._is_active()
		if parent_active:
			value = self._evaluate_condition(args, filename, line_num)
		else:
			value = False
		self._if_stack.append(_IfState(active=value, seen_true=value))

	def _handle_elif(self, args: str, filename: str, line_num: int) -> None:
		"""Handle #elif directive."""
		if not self._if_stack:
			raise PreprocessorError("#elif without matching #if", filename, line_num)
		state = self._if_stack[-1]
		if state.is_else:
			raise PreprocessorError("#elif after #else", filename, line_num)

		# Check if parent context is active
		parent_active = all(s.active for s in self._if_stack[:-1])
		if parent_active and not state.seen_true:
			value = self._evaluate_condition(args, filename, line_num)
			state.active = value
			if value:
				state.seen_true = True
		else:
			state.active = False

	def _handle_else(self, filename: str, line_num: int) -> None:
		"""Handle #else directive."""
		if not self._if_stack:
			raise PreprocessorError("#else without matching #if", filename, line_num)
		state = self._if_stack[-1]
		if state.is_else:
			raise PreprocessorError("Duplicate #else", filename, line_num)

		parent_active = all(s.active for s in self._if_stack[:-1])
		state.active = parent_active and not state.seen_true
		state.is_else = True

	def _handle_endif(self, filename: str, line_num: int) -> None:
		"""Handle #endif directive."""
		if not self._if_stack:
			raise PreprocessorError("#endif without matching #if", filename, line_num)
		self._if_stack.pop()

	def _evaluate_condition(self, expr: str, filename: str, line_num: int) -> bool:
		"""Evaluate a preprocessor conditional expression."""
		# Handle defined() operator
		expr = self._expand_defined(expr)

		# Expand macros in the expression
		expr = self._expand_macros(expr, filename, line_num)

		# Replace any remaining identifiers with 0 (standard C behavior)
		# Preserve Python keywords needed for eval (and, or, not)
		_EVAL_KEYWORDS = {"and", "or", "not", "True", "False"}
		expr = re.sub(
			r"\b[A-Za-z_]\w*\b",
			lambda m: m.group(0) if m.group(0) in _EVAL_KEYWORDS else "0",
			expr,
		)

		# Evaluate the expression
		expr = expr.strip()
		if not expr:
			raise PreprocessorError("Empty #if expression", filename, line_num)

		try:
			# Support C-style logical/bitwise operators
			# Replace C operators that differ from Python
			py_expr = expr
			# Handle character literals like 'A'
			py_expr = re.sub(r"'\\?(.)'", lambda m: str(ord(m.group(0)[1:-1].encode().decode("unicode_escape"))), py_expr)
			result = eval(py_expr, {"__builtins__": {}}, {})  # noqa: S307
			return bool(result)
		except Exception as e:
			raise PreprocessorError(f"Cannot evaluate #if expression: {expr!r}", filename, line_num) from e

	def _expand_defined(self, expr: str) -> str:
		"""Expand defined() and defined NAME in expressions."""
		# defined(NAME)
		expr = re.sub(
			r"\bdefined\s*\(\s*(\w+)\s*\)",
			lambda m: "1" if m.group(1) in self.macros else "0",
			expr,
		)
		# defined NAME
		expr = re.sub(
			r"\bdefined\s+(\w+)",
			lambda m: "1" if m.group(1) in self.macros else "0",
			expr,
		)
		return expr

	def _handle_include(self, args: str, filename: str, line_num: int) -> str:
		"""Handle #include directive."""
		args = args.strip()

		# Expand macros in include argument if not a direct string
		if not (args.startswith('"') or args.startswith("<")):
			args = self._expand_macros(args, filename, line_num).strip()

		# "file" - quoted form
		match = re.match(r'"([^"]+)"', args)
		if match:
			include_name = match.group(1)
			resolved = self._resolve_include(include_name, filename, quoted=True)
			if resolved is None:
				raise PreprocessorError(f"Cannot find include file: {include_name!r}", filename, line_num)
			return self._process_include(resolved, filename, line_num)

		# <file> - angle bracket form
		match = re.match(r"<([^>]+)>", args)
		if match:
			include_name = match.group(1)
			resolved = self._resolve_include(include_name, filename, quoted=False)
			if resolved is None:
				raise PreprocessorError(f"Cannot find include file: {include_name!r}", filename, line_num)
			return self._process_include(resolved, filename, line_num)

		raise PreprocessorError(f"Invalid #include syntax: {args!r}", filename, line_num)

	def _resolve_include(self, name: str, current_file: str, quoted: bool) -> str | None:
		"""Resolve an include file path."""
		# If it's an absolute path, check directly
		if os.path.isabs(name):
			if os.path.isfile(name):
				return os.path.abspath(name)
			return None

		# For quoted includes, search relative to current file first
		if quoted and current_file != "<stdin>":
			current_dir = os.path.dirname(os.path.abspath(current_file))
			candidate = os.path.join(current_dir, name)
			if os.path.isfile(candidate):
				return os.path.abspath(candidate)

		# Search include paths
		for path in self.include_paths:
			candidate = os.path.join(path, name)
			if os.path.isfile(candidate):
				return os.path.abspath(candidate)

		return None

	def _process_include(self, filepath: str, parent_file: str, line_num: int) -> str:
		"""Read and preprocess an included file."""
		abs_path = os.path.abspath(filepath)

		if abs_path in self._included_files:
			return ""  # Already included (simple include guard)

		self._included_files.add(abs_path)

		try:
			with open(filepath) as f:
				content = f.read()
		except OSError as e:
			raise PreprocessorError(f"Cannot read include file: {filepath}", parent_file, line_num) from e

		# Save and restore if_stack state (included files shouldn't leak conditionals)
		saved_stack = self._if_stack[:]
		self._if_stack = []

		result = self.process(content, filepath)

		self._if_stack = saved_stack
		return result

	def _expand_macros(self, text: str, filename: str, line_num: int) -> str:
		"""Expand all macros in text, handling nested expansion."""
		return self._expand_macros_impl(text, filename, line_num, expanding=set())

	def _expand_macros_impl(
		self,
		text: str,
		filename: str,
		line_num: int,
		expanding: set[str],
	) -> str:
		"""Internal macro expansion with recursion guard."""
		# Expand predefined macros
		text = text.replace("__LINE__", str(line_num))
		text = text.replace("__FILE__", f'"{filename}"')

		changed = True
		max_iterations = 100
		iteration = 0

		while changed and iteration < max_iterations:
			changed = False
			iteration += 1

			for name, macro in list(self.macros.items()):
				if name in expanding:
					continue

				if macro.params is not None:
					# Function-like macro: look for NAME(args)
					new_text = self._expand_function_macro(text, name, macro, filename, line_num, expanding)
					if new_text != text:
						text = new_text
						changed = True
				else:
					# Object-like macro: simple substitution with word boundary
					pattern = re.compile(r"\b" + re.escape(name) + r"\b")
					if pattern.search(text):
						new_expanding = expanding | {name}
						expanded_body = self._expand_macros_impl(
							macro.body, filename, line_num, new_expanding
						)
						new_text = self._sub_outside_strings(pattern, expanded_body, text)
						if new_text != text:
							text = new_text
							changed = True

		return text

	@staticmethod
	def _sub_outside_strings(pattern: re.Pattern, replacement: str, text: str) -> str:
		"""Apply regex substitution only outside of string/char literals."""
		result: list[str] = []
		i = 0
		length = len(text)

		while i < length:
			ch = text[i]

			if ch == '"':
				# Consume entire string literal
				j = i + 1
				while j < length:
					if text[j] == "\\" and j + 1 < length:
						j += 2
					elif text[j] == '"':
						j += 1
						break
					else:
						j += 1
				result.append(text[i:j])
				i = j
				continue

			if ch == "'":
				# Consume entire char literal
				j = i + 1
				while j < length:
					if text[j] == "\\" and j + 1 < length:
						j += 2
					elif text[j] == "'":
						j += 1
						break
					else:
						j += 1
				result.append(text[i:j])
				i = j
				continue

			# Try to match the pattern at this position
			m = pattern.match(text, i)
			if m:
				# Check word boundary at the start
				result.append(replacement)
				i = m.end()
			else:
				result.append(ch)
				i += 1

		return "".join(result)

	def _expand_function_macro(
		self,
		text: str,
		name: str,
		macro: Macro,
		filename: str,
		line_num: int,
		expanding: set[str],
	) -> str:
		"""Expand a function-like macro invocation in text."""
		result: list[str] = []
		i = 0
		length = len(text)

		while i < length:
			# Look for the macro name followed by (
			match = re.match(r"\b" + re.escape(name) + r"\b", text[i:])
			if not match:
				result.append(text[i])
				i += 1
				continue

			name_end = i + match.end()

			# Skip whitespace after name
			j = name_end
			while j < length and text[j] in " \t":
				j += 1

			if j >= length or text[j] != "(":
				# Not a function call, just the name
				result.append(text[i:name_end])
				i = name_end
				continue

			# Parse arguments
			args, end_pos = self._parse_macro_args(text, j, filename, line_num)

			if args is None:
				result.append(text[i:name_end])
				i = name_end
				continue

			expected = len(macro.params) if macro.params else 0
			if not macro.is_variadic:
				if len(args) != expected:
					# Allow zero args matching zero params
					if not (expected == 0 and len(args) == 1 and args[0].strip() == ""):
						raise PreprocessorError(
							f"Macro {name} expects {expected} arguments, got {len(args)}",
							filename,
							line_num,
						)
					if expected == 0:
						args = []
			else:
				if len(args) < expected:
					raise PreprocessorError(
						f"Macro {name} expects at least {expected} arguments, got {len(args)}",
						filename,
						line_num,
					)

			# Substitute parameters
			body = macro.body

			if macro.params:
				for pi, param in enumerate(macro.params):
					arg_val = args[pi].strip() if pi < len(args) else ""
					body = re.sub(r"\b" + re.escape(param) + r"\b", arg_val, body)

			if macro.is_variadic:
				va_args = ", ".join(a.strip() for a in args[expected:])
				body = body.replace("__VA_ARGS__", va_args)

			# Recursively expand the result
			new_expanding = expanding | {name}
			body = self._expand_macros_impl(body, filename, line_num, new_expanding)

			result.append(body)
			i = end_pos

		return "".join(result)

	def _parse_macro_args(
		self,
		text: str,
		start: int,
		filename: str,
		line_num: int,
	) -> tuple[list[str] | None, int]:
		"""Parse macro arguments from text starting at '('.

		Returns (args_list, position_after_closing_paren) or (None, start) on failure.
		"""
		if start >= len(text) or text[start] != "(":
			return None, start

		depth = 1
		i = start + 1
		args: list[str] = []
		current_arg: list[str] = []

		while i < len(text) and depth > 0:
			ch = text[i]

			if ch == "(":
				depth += 1
				current_arg.append(ch)
			elif ch == ")":
				depth -= 1
				if depth == 0:
					args.append("".join(current_arg))
				else:
					current_arg.append(ch)
			elif ch == "," and depth == 1:
				args.append("".join(current_arg))
				current_arg = []
			elif ch == '"':
				# Skip string literals
				current_arg.append(ch)
				i += 1
				while i < len(text) and text[i] != '"':
					if text[i] == "\\":
						current_arg.append(text[i])
						i += 1
						if i < len(text):
							current_arg.append(text[i])
					else:
						current_arg.append(text[i])
					i += 1
				if i < len(text):
					current_arg.append(text[i])
			elif ch == "'":
				# Skip char literals
				current_arg.append(ch)
				i += 1
				while i < len(text) and text[i] != "'":
					if text[i] == "\\":
						current_arg.append(text[i])
						i += 1
						if i < len(text):
							current_arg.append(text[i])
					else:
						current_arg.append(text[i])
					i += 1
				if i < len(text):
					current_arg.append(text[i])
			else:
				current_arg.append(ch)

			i += 1

		if depth != 0:
			raise PreprocessorError("Unterminated macro argument list", filename, line_num)

		return args, i
