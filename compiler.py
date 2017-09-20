import ast
import inspect
import astor
import textwrap

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
        return super(self, Ref).__init__(name, index)

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
        return super(self, If).__init__(cond, body, elseBody)

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
        body = [self.visit(stmt) for stmt in func.body]
        
        return FuncDef(func.name, [], body)

def Compile(f):
    """'Compile' the function f"""
    # Parse and extract the function definition AST
    fun = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]
    print(astor.dump(fun))
    
    simpleFun = PythonToSimple().visit(fun)
    
    print(astor.dump(simpleFun))
    
    return f


#############
## TEST IT ##
#############

# Define a trivial test program to start
def trivial() -> int:
    return 5

def test_it():
    Compile(trivial)
    
if __name__ == '__main__':
    test_it()

