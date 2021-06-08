# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# type: ignore

import re

# used to convert server value strings into actual python values
pattern_int = re.compile(r"^-?\d+$")
pattern_float = re.compile(r"^-?\d*[.]\d+$")


def parse(text):  # noqa: C901
    """
    This is what amounts to a simple lisp parser for turning the server's
    returned messages into an intermediate format that's easier to deal
    with than the raw (often poorly formatted) text.

    This parses generally, taking any lisp-like string and turning it into a
    list of nested lists, where each nesting indicates a parenthesized
    expression.  holding multiple top-level parenthesized expressions. Ex: "(baz
    0 (foo 1.5))" becomes ['baz', 0, ['foo', 1.5]].
    """

    text = text.decode()  # Added for python 3

    # make sure all of our parenthesis match
    if text.count("(") == text.count(")"):
        # result acts as a stack that holds the strings grouped by nested parens.
        # result will only ever contain one item, the first level of indenting
        # encountered.  this is because the server (hopefully!) only ever sends one
        # message at a time.
        # TODO: make sure that the server only ever sends one message at a time!
        result = []

        # the current level of indentation, used to append chars to correct level
        indent = 0

        # the non-indenting characters we find. these are kept in a buffer until
        # we indent or dedent, and then are added to the current indent level all
        # at once, for efficiency.
        s = []

        # whether we're currently in the middle of parsing a string
        in_string = False

        # the last character seen, None to begin with
        prev_c = None

        for c in text:
            # prevent parsing parens when inside a string (also ignores escaped
            # '"'s as well). doesn't add the quotes so we don't have to recognize
            # that value as a string via a regex.
            if c == '"' and prev_c != "\\":
                in_string = not in_string

            # we only indent/dedent if not in the middle of parsing a string
            elif c == "(" and not in_string:
                # recurse into current level of nesting
                cur = result
                for i in range(indent):
                    cur = cur[-1]

                # add our buffered string onto the previous level, then clear it
                # for the next.
                if len(s) > 0:
                    val = "".join(s)

                    # try to convert our string into a value and append it to our
                    # list.  failing that, simply append it as an attribute name.
                    if pattern_int.match(val):
                        cur.append(int(val))
                    elif pattern_float.match(val):
                        cur.append(float(val))
                    else:
                        cur.append(val)

                    s = []

                # append a new level of nesting to our list
                cur.append([])

                # increase the indent level so we can get back to this level later
                indent += 1

            elif c == ")" and not in_string:
                # append remaining string buffer before dedenting
                if len(s) > 0:
                    cur = result
                    for i in range(indent):
                        cur = cur[-1]

                    val = "".join(s)

                    # try to convert our string into a value and append it to our
                    # list.  failing that, simply append it as an attribute name.
                    if pattern_int.match(val):
                        cur.append(int(val))
                    elif pattern_float.match(val):
                        cur.append(float(val))
                    else:
                        cur.append(val)

                    s = []

                # we finished with one level, so dedent back to the previous one
                indent -= 1

            # append non-space characters to the buffer list. spaces are delimiters
            # for expressions, hence are special.
            elif c != " ":
                # append the current string character to the buffer list.
                s.append(c)

            # we separate expressions by spaces
            elif c == " " and len(s) > 0:
                cur = result
                for i in range(indent):
                    cur = cur[-1]

                val = "".join(s)

                # try to convert our string into a value and append it to our
                # list.  failing that, simply append it as an attribute name.
                if pattern_int.match(val):
                    cur.append(int(val))
                elif pattern_float.match(val):
                    cur.append(float(val))
                else:
                    cur.append(val)

                s = []

            # save the previous character. used to determine if c is escaped
            prev_c = c

        # this returns the first and only message found.  result is a list simply
        # because it makes adding new levels of indentation simpler as it avoids
        # the 'if result is None' corner case that would come up when trying to
        # append the first '('.
        return result[0]
    else:
        return None
