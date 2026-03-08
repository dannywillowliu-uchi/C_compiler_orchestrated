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


_BUILTIN_HEADERS: dict[str, str] = {
	"stdbool.h": "\n".join([
		"#ifndef _STDBOOL_H",
		"#define _STDBOOL_H",
		"#define bool _Bool",
		"#define true 1",
		"#define false 0",
		"#define __bool_true_false_are_defined 1",
		"#endif",
	]),
	"stdarg.h": "\n".join([
		"#ifndef _STDARG_H",
		"#define _STDARG_H",
		"typedef void *va_list;",
		"#endif",
	]),
	"stddef.h": "\n".join([
		"#ifndef _STDDEF_H",
		"#define _STDDEF_H",
		"typedef unsigned long size_t;",
		"typedef long ptrdiff_t;",
		"typedef int wchar_t;",
		"#define NULL ((void *)0)",
		"#define offsetof(type, member) ((unsigned long)&((type *)0)->member)",
		"#endif",
	]),
	"stdint.h": "\n".join([
		"#ifndef _STDINT_H",
		"#define _STDINT_H",
		"typedef signed char int8_t;",
		"typedef short int16_t;",
		"typedef int int32_t;",
		"typedef long long int64_t;",
		"typedef unsigned char uint8_t;",
		"typedef unsigned short uint16_t;",
		"typedef unsigned int uint32_t;",
		"typedef unsigned long long uint64_t;",
		"typedef long intptr_t;",
		"typedef unsigned long uintptr_t;",
		"typedef long long intmax_t;",
		"typedef unsigned long long uintmax_t;",
		"#define INT8_MIN (-128)",
		"#define INT8_MAX 127",
		"#define INT16_MIN (-32768)",
		"#define INT16_MAX 32767",
		"#define INT32_MIN (-2147483647 - 1)",
		"#define INT32_MAX 2147483647",
		"#define INT64_MIN (-9223372036854775807LL - 1)",
		"#define INT64_MAX 9223372036854775807LL",
		"#define UINT8_MAX 255",
		"#define UINT16_MAX 65535",
		"#define UINT32_MAX 4294967295U",
		"#define UINT64_MAX 18446744073709551615ULL",
		"#define INTPTR_MIN (-9223372036854775807L - 1)",
		"#define INTPTR_MAX 9223372036854775807L",
		"#define UINTPTR_MAX 18446744073709551615UL",
		"#define INTMAX_MIN (-9223372036854775807LL - 1)",
		"#define INTMAX_MAX 9223372036854775807LL",
		"#define UINTMAX_MAX 18446744073709551615ULL",
		"#define SIZE_MAX 18446744073709551615UL",
		"#endif",
	]),
	"limits.h": "\n".join([
		"#ifndef _LIMITS_H",
		"#define _LIMITS_H",
		"#define CHAR_BIT 8",
		"#define SCHAR_MIN (-128)",
		"#define SCHAR_MAX 127",
		"#define UCHAR_MAX 255",
		"#define CHAR_MIN (-128)",
		"#define CHAR_MAX 127",
		"#define SHRT_MIN (-32768)",
		"#define SHRT_MAX 32767",
		"#define USHRT_MAX 65535",
		"#define INT_MIN (-2147483647 - 1)",
		"#define INT_MAX 2147483647",
		"#define UINT_MAX 4294967295U",
		"#define LONG_MIN (-9223372036854775807L - 1)",
		"#define LONG_MAX 9223372036854775807L",
		"#define ULONG_MAX 18446744073709551615UL",
		"#define LLONG_MIN (-9223372036854775807LL - 1)",
		"#define LLONG_MAX 9223372036854775807LL",
		"#define ULLONG_MAX 18446744073709551615ULL",
		"#endif",
	]),
	"string.h": "\n".join([
		"#ifndef _STRING_H",
		"#define _STRING_H",
		"typedef unsigned long size_t;",
		"void *memcpy(void *dest, const void *src, size_t n);",
		"void *memmove(void *dest, const void *src, size_t n);",
		"void *memset(void *s, int c, size_t n);",
		"int memcmp(const void *s1, const void *s2, size_t n);",
		"char *strcpy(char *dest, const char *src);",
		"char *strncpy(char *dest, const char *src, size_t n);",
		"int strcmp(const char *s1, const char *s2);",
		"int strncmp(const char *s1, const char *s2, size_t n);",
		"size_t strlen(const char *s);",
		"char *strcat(char *dest, const char *src);",
		"char *strncat(char *dest, const char *src, size_t n);",
		"char *strchr(const char *s, int c);",
		"char *strrchr(const char *s, int c);",
		"char *strstr(const char *haystack, const char *needle);",
		"#define NULL ((void *)0)",
		"#endif",
	]),
	"assert.h": "\n".join([
		"#ifndef _ASSERT_H",
		"#define _ASSERT_H",
		"#define assert(expr) ((void)0)",
		"#endif",
	]),
}


class Preprocessor:
	"""C preprocessor that transforms source text for the lexer."""

	def __init__(
		self,
		include_paths: list[str] | None = None,
		predefined_macros: dict[str, str] | None = None,
	) -> None:
		self.include_paths: list[str] = include_paths or []
		self.macros: dict[str, Macro] = {}
		self.warnings: list[str] = []
		self._if_stack: list[_IfState] = []
		self._included_files: set[str] = set()
		self._pragma_once_files: set[str] = set()
		self._line_offset: int = 0
		self._file_override: str | None = None

		# GCC-compatible predefined type and constant macros
		_gcc_builtins: dict[str, str] = {
			"__SIZE_TYPE__": "unsigned long",
			"__PTRDIFF_TYPE__": "long",
			"__WCHAR_TYPE__": "int",
			"__WINT_TYPE__": "unsigned int",
			"__INT8_TYPE__": "signed char",
			"__UINT8_TYPE__": "unsigned char",
			"__INT16_TYPE__": "short",
			"__UINT16_TYPE__": "unsigned short",
			"__INT32_TYPE__": "int",
			"__UINT32_TYPE__": "unsigned int",
			"__INT64_TYPE__": "long long",
			"__UINT64_TYPE__": "unsigned long long",
			"__INTPTR_TYPE__": "long",
			"__UINTPTR_TYPE__": "unsigned long",
			"__INT_MAX__": "2147483647",
			"__LONG_MAX__": "9223372036854775807L",
			"__LONG_LONG_MAX__": "9223372036854775807LL",
			"__SIZEOF_INT__": "4",
			"__SIZEOF_LONG__": "8",
			"__SIZEOF_LONG_LONG__": "8",
			"__SIZEOF_SHORT__": "2",
			"__SIZEOF_POINTER__": "8",
			"__SIZEOF_FLOAT__": "4",
			"__SIZEOF_DOUBLE__": "8",
			"__CHAR_BIT__": "8",
			"__BYTE_ORDER__": "1234",
			"__ORDER_LITTLE_ENDIAN__": "1234",
			"__ORDER_BIG_ENDIAN__": "4321",
			"__ORDER_PDP_ENDIAN__": "3412",
			"CHAR_BIT": "8",
			# Predefined integer limit macros (available without headers)
			"INT_MAX": "2147483647",
			"INT_MIN": "(-2147483647 - 1)",
			"INT8_MAX": "127",
			"INT8_MIN": "(-128)",
			"INT16_MAX": "32767",
			"INT16_MIN": "(-32768)",
			"INT32_MAX": "2147483647",
			"INT32_MIN": "(-2147483647 - 1)",
			"INT64_MAX": "9223372036854775807LL",
			"INT64_MIN": "(-9223372036854775807LL - 1)",
			"UINT8_MAX": "255",
			"UINT16_MAX": "65535",
			"UINT32_MAX": "4294967295U",
			"UINT64_MAX": "18446744073709551615ULL",
			"INT_WIDTH": "32",
			"LONG_WIDTH": "64",
			"INTPTR_MAX": "9223372036854775807L",
			"INTPTR_MIN": "(-9223372036854775807L - 1)",
		}
		for name, body in _gcc_builtins.items():
			self.macros[name] = Macro(name=name, body=body)

		# C23: bool, true, false are keywords (available without stdbool.h)
		self.macros["bool"] = Macro(name="bool", body="_Bool")
		self.macros["true"] = Macro(name="true", body="1")
		self.macros["false"] = Macro(name="false", body="0")

		# Built-in assert macro (available without #include <assert.h>)
		# Use if-abort pattern to avoid ternary void cast issues
		self.macros["assert"] = Macro(name="assert", body="do { if (!(expr)) abort(); } while(0)", params=["expr"])

		if predefined_macros:
			for name, body in predefined_macros.items():
				self.macros[name] = Macro(name=name, body=body)


	def preprocess(self, source: str, filename: str = "<stdin>") -> str:
		"""Alias for process() for compatibility."""
		return self.process(source, filename)

	def process(self, source: str, filename: str = "<stdin>") -> str:
		"""Process source text and return preprocessed output."""
		self._line_offset = 0
		self._file_override = None
		source = self._strip_comments(source)
		source = self._join_continuation_lines(source)
		lines = source.split("\n")
		output_lines: list[str] = []

		for line_num, line in enumerate(lines, start=1):
			stripped = line.strip()

			if stripped.startswith("#"):
				self._handle_directive(stripped, filename, line_num, output_lines)
			elif self._is_active():
				effective_line = line_num + self._line_offset
				effective_file = self._file_override or filename
				expanded = self._expand_macros(line, effective_file, effective_line)
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
				effective_file = self._file_override or filename
				effective_line = line_num + self._line_offset
				raise PreprocessorError(f"#error {args}", effective_file, effective_line)
			elif directive == "warning":
				self._handle_warning(args, filename, line_num)
				output_lines.append("")
			elif directive == "line":
				self._handle_line_directive(args, filename, line_num)
				output_lines.append("")
			elif directive == "pragma":
				self._handle_pragma(args, filename, line_num)
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

	def _handle_warning(self, args: str, filename: str, line_num: int) -> None:
		"""Handle #warning directive."""
		effective_file = self._file_override or filename
		effective_line = line_num + self._line_offset
		self.warnings.append(f"{effective_file}:{effective_line}: warning: #warning {args}")

	def _handle_line_directive(self, args: str, filename: str, line_num: int) -> None:
		"""Handle #line directive: #line number [\"filename\"]."""
		parts = args.split(None, 1)
		if not parts:
			raise PreprocessorError("Expected line number after #line", filename, line_num)
		try:
			new_line = int(parts[0])
		except ValueError:
			raise PreprocessorError(f"Invalid line number in #line: {parts[0]!r}", filename, line_num)
		# Next source line (line_num + 1) should report as new_line
		self._line_offset = new_line - line_num - 1
		if len(parts) > 1:
			fname = parts[1].strip()
			if fname.startswith('"') and fname.endswith('"'):
				self._file_override = fname[1:-1]
			else:
				raise PreprocessorError(f"Invalid filename in #line: {fname!r}", filename, line_num)

	def _handle_pragma(self, args: str, filename: str, line_num: int) -> None:
		"""Handle #pragma directive."""
		if args.strip() == "once":
			if filename != "<stdin>":
				self._pragma_once_files.add(os.path.abspath(filename))

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
			py_expr = expr
			# Convert C logical operators to Python equivalents
			py_expr = py_expr.replace("&&", " and ")
			py_expr = py_expr.replace("||", " or ")
			# Replace '!' (logical not) but not '!=' (not-equal)
			py_expr = re.sub(r"!(?!=)", " not ", py_expr)
			# Handle character literals like 'A'
			py_expr = re.sub(r"'\\?(.)'", lambda m: str(ord(m.group(0)[1:-1].encode().decode("unicode_escape"))), py_expr)
			# Strip C integer suffixes (ULL, LL, UL, LU, U, L) from numeric literals
			py_expr = re.sub(r"\b(\d+)\s*(?:ULL|LLU|ull|llu|UL|LU|ul|lu|LL|ll|U|u|L|l)\b", r"\1", py_expr)
			# Handle hex literals with suffixes too
			py_expr = re.sub(r"(0[xX][0-9a-fA-F]+)\s*(?:ULL|LLU|ull|llu|UL|LU|ul|lu|LL|ll|U|u|L|l)\b", r"\1", py_expr)
			result = eval(py_expr, {"__builtins__": {}}, {})  # noqa: S307
			return bool(result)
		except SyntaxError:
			# Expression may contain type names or other non-evaluable tokens;
			# treat as false (0) like GCC does for unevaluable expressions
			return False
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

		# Extract include name for built-in header check
		include_name: str | None = None
		match = re.match(r'"([^"]+)"', args)
		if match:
			include_name = match.group(1)
		else:
			match = re.match(r"<([^>]+)>", args)
			if match:
				include_name = match.group(1)

		if include_name is None:
			raise PreprocessorError(f"Invalid #include syntax: {args!r}", filename, line_num)

		# Check for built-in headers first
		if include_name in _BUILTIN_HEADERS:
			builtin_content = _BUILTIN_HEADERS[include_name]
			return self.process(builtin_content, f"<builtin:{include_name}>")

		# "file" - quoted form
		if args.startswith('"'):
			resolved = self._resolve_include(include_name, filename, quoted=True)
			if resolved is None:
				raise PreprocessorError(f"Cannot find include file: {include_name!r}", filename, line_num)
			return self._process_include(resolved, filename, line_num)

		# <file> - angle bracket form
		resolved = self._resolve_include(include_name, filename, quoted=False)
		if resolved is None:
			raise PreprocessorError(f"Cannot find include file: {include_name!r}", filename, line_num)
		return self._process_include(resolved, filename, line_num)

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

		if abs_path in self._pragma_once_files:
			return ""  # Marked with #pragma once

		self._included_files.add(abs_path)

		try:
			with open(filepath) as f:
				content = f.read()
		except OSError as e:
			raise PreprocessorError(f"Cannot read include file: {filepath}", parent_file, line_num) from e

		# Save and restore state (included files shouldn't leak conditionals or line info)
		saved_stack = self._if_stack[:]
		saved_line_offset = self._line_offset
		saved_file_override = self._file_override
		self._if_stack = []

		result = self.process(content, filepath)

		self._if_stack = saved_stack
		self._line_offset = saved_line_offset
		self._file_override = saved_file_override
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

			# Process function-like macros first so they capture raw arguments
			# before object-like macros can expand identifiers inside arguments
			for name, macro in list(self.macros.items()):
				if name in expanding:
					continue
				if macro.params is not None:
					new_text = self._expand_function_macro(text, name, macro, filename, line_num, expanding)
					if new_text != text:
						text = new_text
						changed = True

			for name, macro in list(self.macros.items()):
				if name in expanding:
					continue
				if macro.params is None:
					pattern = re.compile(r"\b" + re.escape(name) + r"\b")
					if pattern.search(text):
						new_expanding = expanding | {name}
						body = self._apply_token_pasting(macro.body)
						expanded_body = self._expand_macros_impl(
							body, filename, line_num, new_expanding
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

	@staticmethod
	def _stringify(arg: str) -> str:
		"""Stringify a macro argument (# operator)."""
		escaped = arg.strip()
		escaped = escaped.replace("\\", "\\\\")
		escaped = escaped.replace('"', '\\"')
		return f'"{escaped}"'

	def _apply_stringification(
		self, body: str, params: list[str], args: list[str]
	) -> str:
		"""Apply # (stringification) operator in macro body."""
		result: list[str] = []
		i = 0
		while i < len(body):
			ch = body[i]
			# Skip string literals
			if ch == '"':
				j = i + 1
				while j < len(body):
					if body[j] == "\\" and j + 1 < len(body):
						j += 2
					elif body[j] == '"':
						j += 1
						break
					else:
						j += 1
				result.append(body[i:j])
				i = j
				continue
			# Skip char literals
			if ch == "'":
				j = i + 1
				while j < len(body):
					if body[j] == "\\" and j + 1 < len(body):
						j += 2
					elif body[j] == "'":
						j += 1
						break
					else:
						j += 1
				result.append(body[i:j])
				i = j
				continue
			if ch == "#":
				# Check for ## (token pasting - leave it alone)
				if i + 1 < len(body) and body[i + 1] == "#":
					result.append("##")
					i += 2
					continue
				# Try stringification: # followed by optional whitespace and param name
				j = i + 1
				while j < len(body) and body[j] in " \t":
					j += 1
				matched = False
				for pi, param in enumerate(params):
					plen = len(param)
					if body[j : j + plen] == param:
						end = j + plen
						if end >= len(body) or not (body[end].isalnum() or body[end] == "_"):
							arg_val = args[pi] if pi < len(args) else ""
							result.append(self._stringify(arg_val))
							i = end
							matched = True
							break
				if not matched:
					result.append("#")
					i += 1
			else:
				result.append(ch)
				i += 1
		return "".join(result)

	@staticmethod
	def _apply_token_pasting(body: str) -> str:
		"""Apply ## (token pasting) operator in macro body."""
		if "##" not in body:
			return body
		result: list[str] = []
		i = 0
		while i < len(body):
			ch = body[i]
			# Skip string literals
			if ch == '"':
				j = i + 1
				while j < len(body):
					if body[j] == "\\" and j + 1 < len(body):
						j += 2
					elif body[j] == '"':
						j += 1
						break
					else:
						j += 1
				result.append(body[i:j])
				i = j
				continue
			# Skip char literals
			if ch == "'":
				j = i + 1
				while j < len(body):
					if body[j] == "\\" and j + 1 < len(body):
						j += 2
					elif body[j] == "'":
						j += 1
						break
					else:
						j += 1
				result.append(body[i:j])
				i = j
				continue
			# Check for ##
			if ch == "#" and i + 1 < len(body) and body[i + 1] == "#":
				# Remove trailing whitespace from result
				while result and result[-1] in " \t":
					result.pop()
				# Skip ## and leading whitespace after it
				i += 2
				while i < len(body) and body[i] in " \t":
					i += 1
				continue
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
			# Guard: the character before position i must not be a word char,
			# otherwise \b at the start of the slice gives a false positive
			# (e.g. matching T inside ST).
			if i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_"):
				result.append(text[i])
				i += 1
				continue
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

			# 1. Apply stringification (#param) before normal substitution
			if macro.params:
				body = self._apply_stringification(body, macro.params, args)

			# 2. Normal parameter substitution
			if macro.params:
				for pi, param in enumerate(macro.params):
					arg_val = args[pi].strip() if pi < len(args) else ""
					body = re.sub(r"\b" + re.escape(param) + r"\b", arg_val, body)

			if macro.is_variadic:
				va_args = ", ".join(a.strip() for a in args[expected:])
				body = body.replace("__VA_ARGS__", va_args)

			# 3. Apply token pasting (##) after substitution
			body = self._apply_token_pasting(body)

			# 4. Recursively expand the result
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
