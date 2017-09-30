import ast
import inspect

import astor
import textwrap

########
## IR ##
########

# There is an internal representation of each constant type:
#   int -> IntConst
#   float -> FloatConst
#   str -> StrConst
#   True, False, ... -> NamedConst
#
# None is represented by the same python NoneType.
#
# We also add a 'List' type to represent indexable types.
#
# The operators (Uop, Bop, Cop) are represented by the classes from the ast
# module:
#   Neg -> ast.USub
#   Not -> ast.Not
#   LE -> ast.LtE
#   GE -> ast.GtE
#   etc.
#
# Others are more obvious:
#   Add -> ast.Add
#   Mul -> ast.Mult
#   LT -> ast.Lt
#   GT -> ast.Gt
#   etc.
#
# The full mapping is encoded in the Interpret function.
"""
Expr = BinOp(Bop op, Expr left, Expr right)
     | CmpOp(Cop op, Expr left, Expr right)
     | UnOp(Uop op, Expr e)
     | Ref(Ref base | StrConst name, Expr? index)
     | FloatConst(float val)
     | IntConst(int val)
     | StrConst(str val)
     | NamedConst(str name)
     | Call(Ref name, Expr* args)

Uop = Neg | Not
Bop = Add | Sub | Mul | Div | Mod | And | Or
Cop =  EQ |  NE |  LT |  GT |  LE | GE

Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Block body, Block? elseBody)
     | For(StrConst var, Expr min, Expr max, Block body, Expr step)
     | Return(Expr val)
     | FuncDef(StrConst name, StrConst* args, Stmt body)
     | Call(Ref name, Expr* args)
"""

## Exprs ##
class BinOp(ast.AST):
    _fields = ['op', 'left', 'right']


class CmpOp(ast.AST):
    _fields = ['op', 'left', 'right']


class UnOp(ast.AST):
    _fields = ['op', 'e']


class Ref(ast.AST):
    _fields = ['name', 'index']

    def __init__(self, name, index=None):
        super().__init__(name, index)


class IntConst(ast.AST):
    _fields = ['val',]


class FloatConst(ast.AST):
    _fields = ['val',]


class StrConst(ast.AST):
    _fields = ['val',]


class NamedConst(ast.AST):
    _fields = ['name',]


# Add a 'List' type so we can^W should handle variables with indices.
class List(ast.AST):
    _fields = ['values',]


## Stmts ##
class Assign(ast.AST):
    _fields = ['ref', 'val']


class Block(ast.AST):
    _fields = ['body',]


class If(ast.AST):
    _fields = ['cond', 'body', 'elseBody']
    
    def __init__(self, cond, body, elseBody=None):
        super().__init__(cond, body, elseBody)


class For(ast.AST):
    _fields = ['var', 'min', 'max', 'body', 'step']

    def __init__(self, var, min, max, body, step, executing=False):
        # We don't want 'executing' to be treated like other IR node fields.
        self.executing = executing
        super().__init__(var, min, max, body, step)


class Return(ast.AST):
    _fields = ['val',]


class FuncDef(ast.AST):
    _fields = ['name', 'args', 'body', 'defaults']

    def __init__(self, name, args, body, defaults):
        # We need to keep track of which scope this function has access to when
        # it it is interpreted. This is a convenient place to do it (but we
        # don't want to mess with the ast.AST interface, so we don't include it
        # in the _fields).
        self.captured_symbols = None
        super().__init__(name, args, body, defaults)


class Call(ast.AST):
    _fields = ['name', 'args']


class UndefinedSymbolError(Exception):
    pass


class PythonToSimple(ast.NodeVisitor):
    """Translate a Python AST to our simplified IR.
    
    TODO: Your first job is to complete this implementation.
    You only need to translate Python constructs which are actually 
    representable in our simplified IR.
    
    As a bonus, try implementing logic to catch non-representable 
    Python AST constructs and raise a `NotImplementedError` when you
    do. Without this, the compiler will just explode with an 
    arbitrary error or generate malformed results. Carefully catching
    and reporting errors is one of the most challenging parts of 
    building a user-friendly compiler.
    """
    def visit_BinOp(self, node):
        return BinOp(op=node.op,
                     left=self.visit(node.left),
                     right=self.visit(node.right))

    def visit_BoolOp(self, node):
        R"""Boolean operations are awkward.

        An expression like "return (e and d or True) or (False and 7) or 3"
        will be parsed according to the logical order of operations into a
        disjunction of three expressions "e and d or True" and "False and 7"
        and "3". Within these the order of evaluation doesn't really matter.
        Within the sub-expressions, we get another ordering. So the tree looks
        like this:

        (e and d or True) or (False and 7) or 3

                          or
                       /       \
                   or           or
                /      \      /    \
               and    True   and    3
             /    \         /    \
            a      d      False   7

        Since each sub-expression is represented by another visit to a BoolOp
        node, we only have to worry about homogenous expressons of all
        conjunctions or of all disjunctions.
 
        So for the above expression, on the first visit, we just have to emit
        this structure:

           or
         /    \
                or
              /    \
        """
        root_op = None
        last_op = None
        for value in node.values[:-1]:
            bool_op = BinOp(op=node.op, left=self.visit(value))
            if last_op:
                last_op.right = bool_op
            else:
                root_op = bool_op
            last_op = bool_op
        last_op.right = self.visit(node.values[-1])

        assert root_op is not None

        return root_op
                

    def visit_Compare(self, node):
        """Compare is awkward.

        https://greentreesnakes.readthedocs.io/en/latest/nodes.html#Compare

        If we get a compound comparison expression, e.g. "1 < a < 10", we
        actually get a list of comparison operators and operands after the
        left-most. If we follow the path of least resistance and walk the list
        backward, we end up with a tree like:

        Expression: 1     <   a   <   10
                             (*,  LT, 10)
                              |
        Tree:       (1,  LT,  a)

        But doing this way means a depth-first post-order traversal will
        evaluate the expression left-to-right.
        """
        op_val_chain = list(zip(node.ops, node.comparators))

        root_op = None
        last_op = None
        for op, right_operand in reversed(op_val_chain):
            cmpop = CmpOp(op=op,
                          right=self.visit(right_operand))
            if last_op:
                last_op.left = cmpop
            else:
                root_op = cmpop
            last_op = cmpop
        last_op.left = self.visit(node.left)

        assert root_op is not None, "No root comparison expresson found."

        return root_op

    def visit_UnaryOp(self, node):
        return UnOp(op=node.op, e=self.visit(node.operand))

    def visit_List(self, node):
        return List(values=[self.visit(v) for v in node.elts])

    def visit_Name(self, node):
        """References in ast are Names."""
        return Ref(name=StrConst(node.id), index=None)

    def visit_Subscript(self, node):
        base_ref = self.visit(node.value)
        # If the base reference is itself a subscripted value, then we need to
        # add a level of indirection.
        if base_ref.index is not None:
            ref = Ref(name=base_ref)
        else:
            ref = base_ref
        # TODO(aryap): We don't yet support ast.Slices.
        assert isinstance(node.slice, ast.Index), (
            "Unexpected Subscript slice type: %s. It must be 'index'." % type(
                node.slice))
        ref.index = self.visit(node.slice)
        return ref

    def visit_Index(self, node):
        """And I quote: "Simple subscripting with a single value" """
        return self.visit(node.value)

    def visit_NameConstant(self, node):
        # node.value stores True as True. That seems self-referential to me. So
        # I shall store it as a string. Then it can be converted back to
        # whatever the literal True actually means.
        return NamedConst(str(node.value))

    def visit_Num(self, node):
        if isinstance(node.n, int):
            return IntConst(val=node.n)
        elif isinstance(node.n, float):
            return FloatConst(val=node.n)
        else:
            raise NotImplementedError(
                "Missing support for Num type: %s" % type(node.n))

    def visit_Str(self, node):
        return StrConst(node.s)

    def visit_For(self, node):
        # NOTE(aryap): Our simplified IR seems to omit 'break' statements
        # (likewise continue statements) so the 'orelse' block doesn't make
        # much sense.
        body_block = Block(body=[
            self.visit(stmt) for stmt in node.body + node.orelse])

        # NOTE(aryap): For and Assign can assign to multiple variables, but
        # omit that functionality in our IR too.

        # We have to reduce the iterator called in the Python for loop to a
        # simplified range bounded by min and max values (iterators will be
        # over [min, max). To that end we must understand three different
        # signatures for 'range()':
        #
        # range(stop)           var from 0 to less than stop
        # range(start, stop)    var from start to less than stop
        # range(start, stop, step)  var from start to less than stop in
        #                           incrementes of size step
        start = None
        final = IntConst(0)
        fn_call = node.iter
        if (isinstance(fn_call, ast.Call) and
            isinstance(fn_call.func, ast.Name) and
            fn_call.func.id == "range"):
            # Truncate arguments after the second (we already dealt with 'step')
            args = fn_call.args[:2]

            for arg in args:
                # Shift values down.
                start = final
                final = self.visit(arg)

        step = (self.visit(fn_call.args[2])
                if len(fn_call.args) == 3 else IntConst(1))

        # If we couldn't extract bounds from the for loop, we probably did
        # something wrong.
        assert start is not None, (
            "'for'-loop malformed: could not extract bounds")

        return For(var=self.visit(node.target),
                   min=start,
                   max=final,
                   body=body_block,
                   step=step)

    def visit_Call(self, node):
        return Call(name=self.visit(node.func), args=[
            self.visit(arg) for arg in node.args])
    
    def visit_Return(self, node):
        return Return(self.visit(node.value))
    
    def visit_FunctionDef(self, func):
        assert isinstance(func.body, list)
        statements = [self.visit(stmt) for stmt in func.body]
        body = Block(body=statements) if statements else None
        
        assert func.args is not None, "Function has no 'arguments' object."
        arguments = [self.visit(arg) for arg in func.args.args]
        defaults = [self.visit(default) for default in func.args.defaults]

        return FuncDef(StrConst(func.name), arguments, body, defaults)

    # Almost consistent.
    def visit_arg(self, node):
        return StrConst(val=node.arg)

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "Only single-assignment supported."
        return Assign(ref=self.visit(node.targets[0]),
                      val=self.visit(node.value))

    def visit_Expr(self, node):
        """Wrapper for function calls that don't return anything.

        https://greentreesnakes.readthedocs.io/en/latest/nodes.html#expressions

        Also used to, for example, represent doc strings. We'll only include
        'Call's.

        NOTE(aryap): For debugging it's useful to to remove this so that we
        can make calls to 'print' when interpreting with the regular python
        engine.
        """
        if isinstance(node.value, ast.Call):
            return self.visit(node.value)
        return None

    def visit_If(self, node):
        body_block = Block(body=[self.visit(stmt) for stmt in node.body])
        else_block = Block(body=[self.visit(stmt) for stmt in node.orelse])

        return If(cond=self.visit(node.test),
                  body=body_block,
                  elseBody=else_block)

    def generic_visit(self, node):
        """Catch unsupported node types."""
        raise NotImplementedError(
            "No way to visit nodes of type: %s" % type(node))


# Build a visitor to evaluate Exprs, using the symbol table to look up variable
# definitions.
class EvalExpr(ast.NodeVisitor):
    def __init__(self, symbolTable):
        self.syms = symbolTable
    
    def visit_IntConst(self, node):
        return node.val

    def visit_FloatConst(self, node):
        return node.val

    def visit_StrConst(self, node):
        return node.val

    def visit_NamedConst(self, node):
        if node.name == "True":
            return True
        elif node.name == "False":
            return False
        elif node.name == "None":
            return None
        else:
            raise NotImplementedError(
                "Unknown NamedConst val: %s" % node.name)

    def visit_List(self, node):
        return [self.visit(element) for element in node.values]

    def visit_Block(self, node):
        """Visit statements in the block in order, one at a time."""
        for statement in node.body:
            self.visit(statement)

    def CheckIndex(self, symbol, target, index):
        assert isinstance(index, int), (
            "Only integer indices are supported.")
        # We expect a list.
        assert isinstance(target, list), (
            "Symbol '%s' with index '%d' implied a list but didn't find"
            " one." % (symbol, index))
        assert 0 <= index < len(target), (
            "Index '%d' not in range [0, %d)" % (index, len(target)))

    def visit_Ref(self, node):
        """Read a variable's value."""
        if isinstance(node.name, Ref):
            target = self.visit(node.name)
            symbol = self.visit(node.name.name)
        else:
            symbol = node.name.val
            if symbol not in self.syms:
                raise UndefinedSymbolError("'%s' is undefined" % symbol)
            target = self.syms[symbol]

        if node.index is not None:
            index = self.visit(node.index)
            self.CheckIndex(str(symbol), target, index)
            return target[index]
        # Not an indexed variable.
        return target

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.And):
            return left and right
        if isinstance(op, ast.Or):
            return left or right
        raise NotImplementedError("Unknown BinOp op type: %s" % type(op))

    def visit_CmpOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        raise NotImplementedError("Unknown CmpOp op type: %s" % type(op))

    def visit_UnOp(self, node):
        in_value = self.visit(node.e)
        op = node.op
        if isinstance(op, ast.USub):
            return -in_value
        if isinstance(op, ast.Not):
            return not in_value
        raise NotImplementedError(
            "Unknown UnOp op type: %s" % type(op))

    def visit_Call(self, node):
        # Evaluate the arguments, look up the function, then call Interpret
        # on the FuncDef with the given arguments. (Now I'm worried about
        # performance.)
        args = [self.visit(arg) for arg in node.args]
        target = self.visit(node.name)
        return Interpret(target, *args)

    def generic_visit(self, node):
        raise NotImplementedError(
            "EvalExpr does not understand %s" % type(node))


def Interpret(ir, *args):
    assert isinstance(ir, FuncDef)
    # Initialize a symbol table, to store variable => value bindings
    # This doesn't yield perfectly matching semantics for how python inherits
    # scopes, but it's close enough. E.g. updates to inherited variables in
    # inner functions won't be visible after the function finishes.
    syms = ir.captured_symbols.copy() if ir.captured_symbols else {}

    # Initialise the EvalExpr make sense of IR expressions.
    evaluator = EvalExpr(syms)

    # A conveniently-sized list of default arguments.
    defaults = [None for i in range(len(ir.args))]
    for i in range(len(ir.defaults)):
        defaults[-(i + 1)] = evaluator.visit(ir.defaults[i])

    # Seed the symbol table with the function's arguments, setting default
    # values if available. Args are given to us in the order they are declared
    # in the signature, we hope.
    for i, arg in enumerate(ir.args):
        syms[arg.val] = args[i] if i < len(args) else defaults[i]

    # TODO(aryap): It would be nice to include line number information when
    # parsing the IR so that we can yield info about source code errors when
    # interpreting.

    stack = []
    stack.append(ir.body)
    while stack:
        statement = stack.pop()
        if not statement:
            # There might be spurious Nones as stand-ins for unimplemented code.
            continue

        assert isinstance(statement, ast.AST), (
            "statement is of type: %s" % type(statement))

        if isinstance(statement, Return):
            return evaluator.visit(statement.val)

        if isinstance(statement, Call):
            evaluator.visit(statement)

        elif isinstance(statement, Assign):
            symbol = statement.ref.name.val
            value = evaluator.visit(statement.val)

            if statement.ref.index is not None:
                index = evaluator.visit(statement.ref.index)
                target = syms[symbol]
                evaluator.CheckIndex(symbol, target, index)
                syms[symbol][index] = value
            else:
                syms[symbol] = value

        elif isinstance(statement, Block):
            # Unroll the block.
            for inner in reversed(statement.body):
                stack.append(inner)

        elif isinstance(statement, If):
            # If the condition is true, we add the body for execution. If it's
            # false, we add the elseBody.
            condition = evaluator.visit(statement.cond)
            stack.append(statement.body if condition else statement.elseBody)

        elif isinstance(statement, For):
            # Queue the for-loop body if the value of the index variable is not
            # yet equal to the max. Overwrite it even if it exists. Onto the
            # stack, in order, push 1) the for loop again, 2) a new expression
            # to increment the loop variable by 1 (which is fine under our
            # assumption that all for-loops are over ranges) and 3) the
            # for-loop body. The interpreter will subsequently process the
            # body, the increment operation, and finally the test to repeat the
            # code or not.
            #
            # NOTE(aryap): Python's range() doesn't behave as in C/C++. In
            # Python, at loop exit, the index variable's value has not reached
            # the invalid state. i.e. given:
            #   for in in range(5):
            #       ...
            #   # i == 4
            # i remains at value 4 after the loop, not 5, as it would in C++:
            #   int i = 0;
            #   for (; i < 5; ++i) {}
            #   // i == 5
            #
            # We mimick this behaviour by adding an 'undo' statement to reduce
            # the index variable by one step. This avoids a bunch of conditional
            # logic to avoid updating the symbol table if the test fails.
            index_symbol = statement.var.name.val
            if not statement.executing:
                index_value = evaluator.visit(statement.min)
                if index_value >= evaluator.visit(statement.max):
                    # Corner case: loop is done before it ever executes. Python
                    # behaviour is to not create the index variable in this
                    # case.
                    continue
                syms[index_symbol] = evaluator.visit(statement.min)
                statement.executing = True
            else:
                index_value = syms[index_symbol]
                if index_value >= evaluator.visit(statement.max):
                    # This is a bit of a hack. It means we can re-use the same
                    # loop object when there's an outer loop and re-initialise
                    # the index variable.
                    statement.executing = False
                    # Undo the last step to mimic 'range' behaviour (see NOTE).
                    stack.append(Assign(ref=statement.var, val=BinOp(
                        ast.Sub(), statement.var, statement.step)))
                    # This loop is done.
                    continue

            stack.append(statement)
            stack.append(Assign(ref=statement.var, val=BinOp(
                ast.Add(), statement.var, statement.step)))
            stack.append(statement.body)

        elif isinstance(statement, FuncDef):
            # Functions shall be stored in the symbols table so that they can
            # be interpreted when called.
            function_name = evaluator.visit(statement.name)
            if function_name in syms:
                print("WARNING: %s redefined." % function_name)
            statement.captured_symbols = syms
            syms[function_name] = statement

        else:
            raise NotImplementedError(
                "Interpreter still doesn't understand: %s" % type(statement))

    # No return statement in body, hence return value is None.
    return None


def BuildIR(f):
    # Parse and extract the function definition AST
    parsed = ast.parse(inspect.getsource(f)).body[0]

    # NOTE(aryap): These are extremely useful but make tests slow; leaving
    # commented (which is bad practice, I know I know).
    # print("Python AST:\n{}\n".format(astor.dump(parsed)))
    
    simple_ir = PythonToSimple().visit(parsed)
    
    # NOTE(aryap): As above.
    # print("Simple IR:\n{}\n".format(astor.dump(simple_ir)))
    return simple_ir


def Compile(f):
    """'Compile' the function f"""
    simple_ir = BuildIR(f)
    # package up our generated simple IR in a 
    def run(*args):
        return Interpret(simple_ir, *args)
    return run

@Compile
def LessTrivial(c, d):
    a = c * d
    for i in range(5):
        a = 12 * a
        b = 3.14 * a * a
    if a < 36:
        if a < 50:
            return False
        return True
    else:
        if a > 50:
            if a > 75:
                return 9000
            return 5000
        return "Something else"

if __name__ == '__main__':
    print("This is a basic smoke test. Please run compiler_test.py instead.")
    print(LessTrivial(.5, .5))
    print("Please run compiler_test.py instead.")
