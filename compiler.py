import ast
import inspect
import astor

########
## IR ##
########
"""
Expr = BinOp(Bop op, Expr left, Expr right)
     | CmpOp(Cop op, Expr left, Expr right)
     | UnOp(Uop op, Expr e)
     | Ref(Str name, Expr? index)
     | FloatConst(float val)
     | IntConst(int val)

Uop = Neg | Not
Bop = Add | Sub | Mul | Div | Mod | And | Or
Cop =  EQ |  NE |  LT |  GT |  LE | GE

Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Stmt body, Stmt? elseBody)
     | For(Str var, Expr min, Expr max, Stmt body)
     | Return(Expr val)
	 | FuncDef(Str name, Str* args, Stmt body)
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

## Stmts ##
class Assign(ast.AST):
    _fields = ['ref', 'val']

class Block(ast.AST):
    _fields = ['body',]

class If(ast.AST):
    _fields = ['cond', 'body', 'elseBody']
    
    def __init__(self, cond, body, elseBody=None):
        return super().__init__(cond, body, elseBody)

class For(ast.AST):
    _fields = ['var', 'min', 'max', 'body']

class Return(ast.AST):
    _fields = ['val',]

class FuncDef(ast.AST):
    _fields = ['name', 'args', 'body']

class PythonToSimple(ast.NodeVisitor):
    """
    Translate a Python AST to our simplified IR.
    
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
    def visit_Num(self, node):
        if isinstance(node.n, int):
            return IntConst(val=node.n)
        else:
            raise NotImplementedError("TODO: implement me!")
    
    def visit_Return(self, node):
        return Return(self.visit(node.value))
    
    def visit_FunctionDef(self, func):
        assert len(func.args.args) == 0 # TODO: handle functions with arguments 

        assert isinstance(func.body, list)
        assert len(func.body) == 1 # TODO: handle function bodies with >1 Stmt
        body = self.visit(func.body[0])
        
        return FuncDef(func.name, [], body)

def Interpret(ir, *args):
    assert isinstance(ir, FuncDef)
    assert len(args) == 0 # TODO: you should handle functions with arguments
    
    # Initialize a symbol table, to store variable => value bindings
    # TODO: fill this with the function arguments to start
    syms = {}
    
    # Build a visitor to evaluate Exprs, using the symbol table to look up
    # variable definitions
    class EvalExpr(ast.NodeVisitor):
        def __init__(self, symbolTable):
            self.syms = symbolTable
        
        def visit_IntConst(self, node):
            return node.val
    
    evaluator = EvalExpr(syms)
    
    # TODO: you will probably need to track more than just a single current
    #       statement to deal with Blocks and nesting.
    stmt = ir.body
    while True:
        assert isinstance(stmt, ast.AST)
        if isinstance(stmt, Return):
            return evaluator.visit(stmt.val)
        else:
            raise NotImplementedError("TODO: add support for the full IR")

def Compile(f):
    """'Compile' the function f"""
    # Parse and extract the function definition AST
    fun = ast.parse(inspect.getsource(f)).body[0]
    print("Python AST:\n{}\n".format(astor.dump(fun)))
    
    simpleFun = PythonToSimple().visit(fun)
    
    print("Simple IR:\n{}\n".format(astor.dump(simpleFun)))
    
    # package up our generated simple IR in a 
    def run(*args):
        return Interpret(simpleFun, *args)
    
    return run


#############
## TEST IT ##
#############

# Define a trivial test program to start
def trivial() -> int:
    return 5

def test_it():
    trivialInterpreted = Compile(trivial)
    # run the original and our version, checking that their output matches:
    assert trivial() == trivialInterpreted()
    
    # TODO: add more of your own tests which exercise the functionality
    #       of your completed implementation
    
if __name__ == '__main__':
    test_it()
