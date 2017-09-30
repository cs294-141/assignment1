# Submission Notes

There doesn't seem to be any requirement for comments, documentation or even a report, so I'll add a few notes here:

- Please run the tests with `python3.6 compiler_test.py`.

You should hopefully see what I see:

```
----------------------------------------------------------------------
Ran 4007 tests in 71.459s

OK
```

- I took a few liberties with our intermediate IR to come up with a more complete subset of the language. I felt gross at points having to shuffle between Python native types, `ast` types, and our types. So for example the simple IR also managed `StrConst`s and `NamedConst`s. It also handles array indexing (even multidimensionally).
- I had to add a bastardised notion of scoping to get Fibonacci to work. Sorry, Python.
- I ran out of creative steam for the tests so resorted to brute force. It actually seems feasible to generate the minimum combination of types the tree can support (limiting multiplicities to 1), but that didn't feel like a good test. Instead, basic operations are tested, as are the range of small "simple" functions that can combine `if`s and `for`s.

# CS294-141: Building your first compiler
For the next few weeks, we will be working through a project to build a basic compiler. In doing this, we will gain experience with several useful tools for practical compiler development (namely, the [Python AST library][pyast] for embedded front-end development, and [LLVM][2] for back-end code generation). Along the way we will also encounter common data structures, patterns, and idioms in building compilers and DSLs.

## What are we going to build?
We will build a simple compiler from a subset of Python to optimized machine code. If you are familiar with [Numba][3], you can think of this assignment as building our own toy Numba JIT. With that, we will be able to write code like the following:

```
@Compile
def square(x : int) -> int:
    return x*x
```

Using the `@Compile` decorator will invoke our compiler on the `square` function, parsing, analyzing, JIT compiling, and ultimately replacing it with native machine code.

### Background
If you are not very familiar with Python, the `@Compile` syntax may look magic or unclear. Don't worry, like most things in Python, it's actually very simple: `@Compile` is just a ["decorator"][4] we will define to package up our functionality. Decorators are just syntactic sugar for applying a higher-order function (a function that takes and returns other functions, in this case `Compile`) to another function (in this case, `square`) immediately after its definition, and replacing that definition with the result of the higher-order function applied to it. In other words, this is just syntactic sugar for:

```
# define square just like any other Python function:
def square(x : int) -> int:
    return x*x

# replace the definition of square with a compiled version of itself:
square = Compile(square)
```

If you are familiar with Python and that code still looks strange to you, it's probably because of the type annotations. Yes, this is real, vanilla Python, it's just using a very new addition to the syntax called ["type hints"][5] (we will be using Python 3.6, which is the first version where type hints are officially standard).

# PART 1: THE FRONT-END<br/><span style="font-size: 75%">Or, parsing Python into our own IR</span>
To begin, we need a way to turn actual Python code into something we can work with. This will be our job for the first assignment.

Generally, the front-end of a compiler is responsible for mapping from raw input into an [Abstract Syntax Tree (AST)][6], an Intermediate Representation (IR) which corresponds more cleanly to the logical level at which we want to think about user code. (The "abstract" name distinguishes an AST from a literal parse tree. An AST is generally simplified and normalized beyond the raw output of a parser in external languages, or from operator traces or other information in embedded languages.)

Python is a nice platform for building language extensions because the standard library includes rich tools for [parsing][pyast], [representing][8], and [manipulating][9] the complete Python syntax within the language itself. Using this, Python programs can relatively represent and manipulate *their own* ASTs.

For our compiler, we are only going to worry about a simplified subset of Python. Specifically, we're only going to compile constructs which trivially map into the following simple IR:

```
Expr = BinOp(Bop op, Expr left, Expr right)
     | CmpOp(Cop op, Expr left, Expr right)
     | UnOp(Uop op, Expr e)
     | Ref(Str name, Expr? index)
     | FloatConst(float val)
     | IntConst(int val)

Uop = Neg | Not
Bop = Add | Sub | Mul | Div | Mod | And | Or
Cop = EQ | NE | LT | GT | LE | GE

Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Stmt body, Stmt? elseBody)
     | For(Str var, Expr min, Expr max, Stmt body)
     | Return(Expr val)
	 | FuncDef(Str name, Str* args, Stmt body)
```

This is a description of our intermediate representation as an [algebraic data type][10]. You should read this as "An Expr[ession] can be either a Bin[ary]Op, containing an op and left and right child Exprs, or a CmpOp, containing..., or an Int[eger]Const[ant] containing an int value." The `?` suffix means that a field is optional (so a `Ref[erence]` has an optional `index` expression for when it is referencing an element of an array rather than a scalar variable), while a `*` suffix means that it is a list of 0 or more items of that type. (You will see similar notation, written in an actual data structure generation DSL called ASDL, in the [official documentation of the Python AST][11].) ADTs are an especially useful notation when describing tree-structured data, where each tree node can be one of many types, as is usually the case in compiler IRs.

This IR is similar in structure to many imperative languages like C or Python: it has separate notions of "expressions" (trees of basic math operations, reads from variables, and constants) and "statements" (the top-level, sequentially-ordered operations which are delimited by semicolons in C or separate lines in Python). A statement can `Assign` an expression to a variable reference, `Return` a  value, perform an `If`/else branch or a `For` loop over more statements, or encapsulate a list of potentially many statements in a `Block` (like the contents of a pair of braces `{ }` in C).

The corresponding data structures are implemented in the code as a set of classes based on the Python base `ast.AST` node type. We use this not because we're going to intermingle our IR nodes with the original Python AST nodes, but because `ast.AST` embodies a nice, simple design pattern for representing these kinds of ADTs by simply declaring the list of fields we want each class to have, as well as utility libraries for recursively traversing the AST using the [visitor pattern][12], namely the [`ast.NodeVisitor`][13]. They also integrate naturally with the [astor][14] AST utility library, which includes useful utilities for pretty-printing Python ASTs (`astor.dump`), and re-generating Python code from an AST.

Here's a quick overview of what you can do with IR nodes based on `ast.AST`:

```
>>> import ast, astor
>>> class MyIRNode(ast.AST):
...   _fields = ('foo', 'bar')
...   

# construct with no fields assigned yet
# fields become attributes of the node object
>>> a = MyIRNode()
>>> a.foo = 'hi'
>>> a.bar = 'bye'

# use astor to pretty-print the IR
>>> astor.dump(a)
"MyIRNode(foo='hi', bar='bye')"

>>> b = MyIRNode('left', 'right') # construct fields by positional order
>>> c = MyIRNode(foo=b, bar=a)    # construct fields by name

# c has a and b as child nodes:
>>> astor.dump(c)
"MyIRNode(foo=MyIRNode(foo='left', bar='right'), bar=MyIRNode(foo='hi', bar='bye'))"

# Build a custom IR visitor.
# For each node they visit, NodeVisitors try to dispatch to a method named
# visit_[NodeClassName]. If it doesn't exist, they fall back to generic_visit.
>>> class MyIRVisitor(ast.NodeVisitor):
...   def visit_MyIRNode(self, node):
...     return str.format("MyIRNode(foo={}, bar={})",
...                       self.visit(node.foo),
...                       self.visit(node.bar))
...   def generic_visit(self, node):
...     return str(node)
...     
>>> MyIRVisitor().visit(c)
'MyIRNode(foo=MyIRNode(foo=left, bar=right), bar=MyIRNode(foo=hi, bar=bye))'

```

## Setup
- Install Python 3.6. I highly recommend [Anaconda][15], on which I will build a distribution for later parts of this project, but for part 1 most versions of Python 3 should work fine.
- Install astor: `pip install astor`.
- Fork and clone your own copy of this repository.
- If you're rather new to Python and aren’t already familiar with it, I highly recommend [IPython](http://ipython.org): `conda install ipython` or `pip install ipython` to be sure you have it. It includes both the now-popular [Jupyter Notebook][16] interactive web interface, and just a much better [REPL][17] for interactively writing and testing Python on the command line. Run `ipython` in your shell. Experience the wonders of tab completion, syntax highlighting, shell history, etc. Run `ipython notebook` to get the web notebook interface.

## Step 1: Converting Python ASTs into our IR
Our first task is to translate Python code into our simplified IR. We will do this by building an `ast.NodeVisitor` to recursively walk over a piece of Python AST and construct the corresponding simplified IR. The skeleton for this is defined in the `PythonToSimple` class in the starter code.

Running `python compiler.py` executes this on a trivial function using the simple test code at the bottom. The starter code implements just the bare minimum necessary to translate the function:

```
def trivial():
    return 5
```

Your first job is to complete the `PythonToSimple` visitor to translate any reasonably expressible Python.

I haven't yet provided any additional tests, but you can extend this arbitrarily to write and test your own richer examples.

To understand the Python AST, and how to build `ast.NodeVisitor`s, I highly recommend not only the [official module documentation][pyast], but also the [GreenTreeSnakes unofficial documentation][greentreesnakes].

Bonus: add explicit `NotImplemented` exceptions when attempting to translate non-translatable Python IR.

### _Clarifications:_
For loops in our simple IR take a different, simpler form than Python's for loop construct. (This is motivated both by simplicity, and by building something which will be useful once we are generating native code.) You should focus on lifting Python for loops of the form `for var in range([expr,] expr)`. `help(range)` if you're not very familiar with Python.

Also, a design suggestion: the initial starter code release translated the list of statements in a Python `FunctionDef` AST node's body field into a `list` of statements in our `FuncDef`'s body field. However, it is likely to be cleaner and more uniform to only allow a single statement node to go in the body field. Lists of Stmts should only be contained within a `Block` node. Because the starter code does not translate Blocks, it now exclusively handles single-statement function bodies, and directly generates a `Return` node (with no enclosing list) when translating the `trivial` test function.

## Step 2: Executing our IR
To test this, and to exercise our skills working with tree-structured IRs, we will also build a simple interpreter for our IR, which we can execute and compare with the original. The skeleton for this is defined in the `Interpret` function in the starter code.

As in step 1, the starter code implements the bare minimum necessary to interpret the simple IR for the `trivial` test function. The skeleton includes another `ast.NodeVisitor` for recursively evaluating expressions, and an interpreter loop which runs statements until it reaches a `Return` node, whose value it then returns.

While implementing evaluation handlers for all of the remaining parts of the IR, you will need to think about two things:

1. How will you store the bindings between variable names and values? This is a job usually handled by a "symbol table," as alluded to in the starter code. In an interpreter, a symbol table maps names to concrete values, but the same pattern will be useful for tracking metadata about names once we start analyzing and compiling code. (Python's scoping semantics are extremely flat and dynamic, so you should be able to get by with a single, flat dictionary for your symbol table while interpreting or compiling a function.)
2. How will you keep track of where you are in the execution of the program IR? The `EvalExpr` visitor implicitly tracks its location in an expression tree with the recursive call stack, but the main interpreter loop we've started (`while True…`) needs to explicitly keep track of where it is and where it should go next, including while traversing through nested statements like `Block`, `If`, and `For`. You probably need some kind of stack structure. (If you're new to Python, the regular `list` makes a good stack thanks to its `append` and `pop` methods.)

You are free to implement this however you want—you can even tear out the interpreter loop entirely and build something recursive. What matters is that `Interpret(Compile(f), *args)` returns the same result as `f(*args)` for any reasonably expressible Python function `f`.

## Step 3: Testing
Our starter code includes a single trivial test: it `Compile`s the function `trivial`, and then `Interpret`s the result and compares it to the result of running the original Python function. While completing your implementation, you should add (lots of!) your own end-to-end tests like this. You won't be directly evaluated on your tests, but you need to think seriously about them to be sure you exercise all of the supported language features, logic, and design decisions in your implementation!

## Submission
**_This assignment is due by 11:59pm on Friday, 9/29._**

You should submit your code via GitHub Classroom. If you already did work in a manually-created fork, I suggest:

1. Merging the updates from this master repository, if necessary.
2. Copying your `compiler.py` (and anything else you've created) over top of a fresh checkout from the repo created when you join the assignment on GitHub Classroom.

My apologies for the extra steps for those of you who have already started. Let me know if you have any trouble.

[2]:	http://llvm.org
[3]:	https://numba.pydata.org
[4]:	https://www.python.org/dev/peps/pep-0318/
[5]:	https://www.python.org/dev/peps/pep-0484/
[6]:	https://en.wikipedia.org/wiki/Abstract_syntax_tree
[pyast]:	https://docs.python.org/3/library/ast.html
[8]:	https://docs.python.org/3/library/ast.html#ast.AST
[9]:	https://docs.python.org/3/library/ast.html#ast.NodeVisitor
[10]:	https://en.wikipedia.org/wiki/Algebraic_data_type
[11]:	https://docs.python.org/3/library/ast.html#abstract-grammar
[12]:	https://en.wikipedia.org/wiki/Visitor_pattern
[13]:	https://docs.python.org/3/library/ast.html#ast.NodeVisitor
[14]:	https://github.com/berkerpeksag/astor
[15]:	https://www.anaconda.com/download
[16]:	http://jupyter.org
[17]:	https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop
[greentreesnakes]:	https://greentreesnakes.readthedocs.io/en/latest/
