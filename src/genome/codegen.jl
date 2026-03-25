# codegen.jl — Code generation infrastructure for expression-tree genetic programming.
#
# Modified from the original standalone codegen.jl:
# - Added rng::AbstractRNG field to GenState
# - All rand()/randn() calls now use explicit s.rng or rng parameter
# - No use of global RNG anywhere

# Supported:
# - Float32, Int32 and Bool data types, scalar only
# - arbitrary fixed number and types of input variables, output variables, and temp variables
# - while and for loops
# - if-else statments
# - blocks
# - break and continue statements
# - function calls
# - assignment statements

struct FunctionDetails
	name::Symbol
	args::Vector{DataType}
	return_type::DataType
end

Base.:(==)(a::FunctionDetails, b::FunctionDetails) = a.name == b.name && a.args == b.args && a.return_type == b.return_type
Base.isequal(a::FunctionDetails, b::FunctionDetails) = hash(a) == hash(b)
function Base.hash(a::FunctionDetails, h::UInt64)
	h = hash(a.name, h)
	for arg in a.args
		h = hash(arg, h)
	end
	return hash(a.return_type, h)
end

struct FunctionSet
	funcs::Set{FunctionDetails}
end

function add!(fset::FunctionSet, f::Symbol, nargs::Int, input_type::DataType, return_type::DataType)
	push!(fset.funcs, FunctionDetails(f, fill(input_type, nargs), return_type))
end

# Deterministic sampling helpers.
# Dict and Set iteration order in Julia depends on hash values which can differ
# between sessions. To ensure reproducibility from a seed, we sort collections
# into a canonical order before sampling.
_fd_sort_key(fd::FunctionDetails) = (string(fd.name), string(fd.return_type), length(fd.args), string(fd.args))
_sorted_funcs(funcs) = sort!(collect(funcs), by=_fd_sort_key)
_sorted_pairs(d) = sort!(collect(d), by=first)
_sorted_types(types) = sort!(collect(types), by=string)


struct GenState
	rng::AbstractRNG
	statement_types::Vector{Symbol}
	funcs::FunctionSet
	inputs::Dict{Symbol, DataType}
	outputs::Dict{Symbol, DataType}
	temps::Dict{Symbol, DataType}
	used_types::Set{DataType}
	all_vars::Dict{Symbol, DataType}
end

function GenState(rng::AbstractRNG, fset::FunctionSet, inputs::Dict{Symbol, DataType}, outputs::Dict{Symbol, DataType}, num_temps::Int)
	statement_types = [:(=), :call, :for, :while, :if, :block]

	# Bool is always a used type, since it's required for conditional statements.
	used_types = Set([Bool])
	for (k, v) in union(inputs, outputs)
		push!(used_types, v)
	end

	# Create the specified number of temp variables with deterministic names.
	# Using fixed names (not gensym) and sorted type list ensures reproducibility.
	used_types_vec = _sorted_types(used_types)
	temps = Dict([(Symbol("__temp_$i"), rand(rng, used_types_vec)) for i in 1:num_temps])

	# Create convenience collections.
	all_vars = Dict(union(inputs, outputs, temps))

	return GenState(rng, statement_types, fset, inputs, outputs, temps, used_types, all_vars)
end

# Backward-compatible constructor that creates a default RNG.
function GenState(fset::FunctionSet, inputs::Dict{Symbol, DataType}, outputs::Dict{Symbol, DataType}, num_temps::Int)
	GenState(Random.default_rng(), fset, inputs, outputs, num_temps)
end

###################################################################################
# Function-related behavior.
###################################################################################

"""
Return a function with the same argument types and return type as the given function call expression.
"""
function get_similar_random_function(s::GenState, func::Expr)
	@assert  func.head == :call
	return_type = get_rvalue_type(s, func)
	arg_types = [get_rvalue_type(s, a) for a in func.args[2:end]]
	rand(s.rng, _sorted_funcs(f for f in s.funcs.funcs if f.return_type == return_type && length(f.args) == length(arg_types) && all(f.args .== arg_types)))
end

"""
Return all functions with the given name.
"""
function get_functions(s::GenState, name::Symbol)
	_sorted_funcs(fd for fd in s.funcs.funcs if fd.name == name)
end

"""
Return all functions with the given name and argument types.
"""
function get_functions(s::GenState, name::Symbol, argtypes::Vector{DataType})
	_sorted_funcs(fd for fd in s.funcs.funcs if fd.name == name && fd.args == argtypes)
end


###################################################################################
# Value-related behavior.
###################################################################################

function get_lvalue_type(s::GenState, value)
	s.all_vars[value]
end

function get_rvalues_of_type(s::GenState, T::DataType)
	_sorted_pairs(filter(v -> v[2] == T, union(s.inputs, s.temps)))
end

function get_lvalues(s::GenState)
	_sorted_pairs(union(s.outputs, s.temps))
end

function get_lvalues_of_type(s::GenState, T::DataType)
	_sorted_pairs(filter(v -> v[2] == T, union(s.outputs, s.temps)))
end

function perturb_literal(rng::AbstractRNG, expr::Number)
	T = typeof(expr)
	if expr isa Bool
		return T(rand(rng, Bool))
	elseif expr isa Integer
		return expr + T(rand(rng, -2:2))
	else
		return expr * T(1 + 0.2 * randn(rng, T))
	end
end

function get_random_literal(rng::AbstractRNG, T::DataType)
	if T <: Bool
		return T(rand(rng, Bool))
	elseif T <: Integer
		return T(rand(rng, -100:100))
	else
		if rand(rng, Bool)
			return T(exp(5 * randn(rng)))
		else
			return T(randn(rng))
		end
	end
end

function create_random_rvalue(s::GenState, T::DataType)
	available_variables = get_rvalues_of_type(s, T)
	if !isempty(available_variables) && rand(s.rng, Bool)
		r = rand(s.rng, available_variables)[1]
	else
		r = get_random_literal(s.rng, T)
	end
end

function get_rvalue_type(s::GenState, value)
	throw(ArgumentError("Unexpected value type encountered: $(repr(value)). Expected a Symbol, Number or Expr."))
end

function get_rvalue_type(s::GenState, value::Number)
	typeof(value)
end

function get_rvalue_type(s::GenState, value::Symbol)
	s.all_vars[value]
end

function get_rvalue_type(s::GenState, value::Expr)
	if value.head == :call
		arg_types = [get_rvalue_type(s, arg) for arg in value.args[2:end]]
		matching_functions = get_functions(s, value.args[1], arg_types)
		if isempty(matching_functions)
			throw(ErrorException("No functions matching $(value.args[1])($arg_types)"))
		end

		return first(matching_functions).return_type
	else
		throw(ArgumentError("Unexpected expression head: $(value.head)"))
	end
end

###################################################################################
# Expression creation and mutation.
###################################################################################

function create_random_assignment(s::GenState)
	while true
		v = rand(s.rng, get_lvalues(s))
		r = create_random_rvalue(s, v[2])
		# Avoid self-assignment.
		if v[1] != r
			return :($(v[1]) = $r)
			break
		end
	end
end

"""
Wrap an rvalue in a function call, if possible.
"""
function wrap_rvalue(s::GenState, value)
	T = get_rvalue_type(s, value)
	# Get functions with the required return type; if none exist return the value as-is.
	available_funcs = _sorted_funcs(f for f in s.funcs.funcs if f.return_type == T && all(f.args .== T))
	if isempty(available_funcs)
		return value
	end
	f = rand(s.rng, available_funcs)
	if length(f.args) == 1 && f.args[1] == T
		Expr(:call, f.name, value)
	elseif length(f.args) == 2 && all(f.args .== T)
		Expr(:call, f.name, value, create_random_rvalue(s, T))
	end
end

function wrap_or_replace_with_similar_rvalue(s::GenState, expr)
	if rand(s.rng, Bool)
		T = get_rvalue_type(s, expr)
		create_random_rvalue(s, T)
	else
		wrap_rvalue(s, expr)
	end
end

function mutate_assignment!(s::GenState, expr::Expr)
	if rand(s.rng, Bool)
		new_lvalue_info = rand(s.rng, get_lvalues_of_type(s, s.all_vars[expr.args[1]]))
		expr.args[1] = new_lvalue_info[1]
	else
		expr.args[2] = wrap_or_replace_with_similar_rvalue(s, expr.args[2])
	end
	expr
end


"""
Create a random function call which returns the required type.
"""
function create_random_function_call(s::GenState, returnType::DataType)
	valid_funcs = _sorted_funcs(f for f in s.funcs.funcs if f.return_type == returnType)
	if isempty(valid_funcs)
		return create_random_assignment(s)
	end
	f = rand(s.rng, valid_funcs)
	return Expr(:call, f.name, [create_random_rvalue(s, argType) for argType in f.args]...)
end

function mutate_function_call!(s::GenState, expr::Expr)
	if rand(s.rng, Bool)
		expr.args[1] = get_similar_random_function(s, expr).name
	else
		arg = rand(s.rng, 2:length(expr.args))
		argType = get_rvalue_type(s, expr.args[arg])
		expr.args[arg] = create_random_rvalue(s, argType)
	end
	expr
end


###################################################################################
# Mutators for compound statements.
###################################################################################

function mutate_for_loop!(s::GenState, expr::Expr)
	if rand(s.rng, Bool)
		lo = Int32(rand(s.rng, 0:10))
		hi = lo + Int32(rand(s.rng, 1:20))
		iter_var = expr.args[1].args[1]
		expr.args[1] = Expr(:(=), iter_var, Expr(:call, :(:), lo, hi))
	else
		expr.args[2] = create_random_assignment(s)
	end
end

function mutate_while_loop!(s::GenState, expr::Expr)
	if rand(s.rng, Bool)
		expr.args[1] = create_random_rvalue(s, Bool)
	else
		expr.args[2] = create_random_assignment(s)
	end
end

function mutate_if_statement!(s::GenState, expr::Expr)
	r = rand(s.rng, 0:2)
	if r == 0
		expr.args[1] = create_random_rvalue(s, Bool)
	elseif r == 1
		expr.args[2] = create_random_assignment(s)
	else
		expr.args[3] = create_random_assignment(s)
	end
end

function mutate_block!(s::GenState, expr::Expr)
	r = rand(s.rng, 0:2)
	if r == 0 || isempty(expr.args)
		insert!(expr.args, rand(s.rng, 1:length(expr.args)+1), create_random_assignment(s))
	elseif r == 1
		deleteat!(expr.args, rand(s.rng, 1:length(expr.args)))
	else
		expr.args[rand(s.rng, 1:length(expr.args))] = create_random_assignment(s)
	end
end


###################################################################################
# Random creation for compound statements (symmetry with mutators above).
###################################################################################

"""
Create a random for loop that iterates over an Int32 range.
Body is a random block; depth limits nesting recursion.
"""
function create_random_for_loop(s::GenState; depth::Int=3)
	if depth <= 0
		return create_random_assignment(s)
	end
	iter_var = Symbol("_i", rand(s.rng, 1000:9999))
	lo = Int32(rand(s.rng, 0:10))
	hi = lo + Int32(rand(s.rng, 1:20))
	range_expr = Expr(:call, :(:), lo, hi)
	iter_assign = Expr(:(=), iter_var, range_expr)
	body = create_random_block(s; depth=depth-1)
	return Expr(:for, iter_assign, body)
end

"""
Create a random while loop with a Bool-typed condition.
Body is a random block; depth limits nesting recursion.
"""
function create_random_while_loop(s::GenState; depth::Int=3)
	if depth <= 0
		return create_random_assignment(s)
	end
	condition = create_random_rvalue(s, Bool)
	body = create_random_block(s; depth=depth-1)
	return Expr(:while, condition, body)
end

"""
Create a random if-else statement with a Bool-typed condition.
Both branches are random blocks; depth limits nesting recursion.
"""
function create_random_if_statement(s::GenState; depth::Int=3)
	if depth <= 0
		return create_random_assignment(s)
	end
	condition = create_random_rvalue(s, Bool)
	true_branch = create_random_block(s; depth=depth-1)
	false_branch = create_random_block(s; depth=depth-1)
	return Expr(:if, condition, true_branch, false_branch)
end

"""
Create a random block of 1-3 statements.
Depth limits nesting recursion.
"""
function create_random_block(s::GenState; depth::Int=3)
	if depth <= 0
		return Expr(:block, create_random_assignment(s))
	end
	n = rand(s.rng, 1:3)
	stmts = [create_random_statement(s; depth=depth-1) for _ in 1:n]
	return Expr(:block, stmts...)
end


"""
Create a random statement of any supported type.
Depth limits nesting recursion; at depth 0, returns an assignment.
"""
function create_random_statement(s::GenState; depth::Int=3)
	if depth <= 0
		return create_random_assignment(s)
	end

	s_type = rand(s.rng, s.statement_types)

	if s_type == :(=)
		create_random_assignment(s)
	elseif s_type == :call
		create_random_function_call(s, rand(s.rng, _sorted_types(s.used_types)))
	elseif s_type == :for
		create_random_for_loop(s; depth=depth)
	elseif s_type == :while
		create_random_while_loop(s; depth=depth)
	elseif s_type == :if
		create_random_if_statement(s; depth=depth)
	elseif s_type == :block
		create_random_block(s; depth=depth)
	else
		error("Unknown statement type: $s_type")
	end
end


function mutate!(s::GenState, expr::Expr)
	if expr.head == :(=)
		mutate_assignment!(s, expr)
	elseif expr.head == :call
		mutate_function_call!(s, expr)
	elseif expr.head == :for
		mutate_for_loop!(s, expr)
	elseif expr.head == :while
		mutate_while_loop!(s, expr)
	elseif expr.head == :if
		mutate_if_statement!(s, expr)
	elseif expr.head == :block
		mutate_block!(s, expr)
	else
		throw(ErrorException("Unsupported expression type $(expr.head)"))
	end

	return expr
end

###################################################################################
# Function construction and harness.
###################################################################################

"""
Construct a Julia function from a signature, body expressions, return expression,
and return type. Evaluates the function into the current scope via @eval.

The return_expr may or may not be wrapped in :return; if it is, the value is
extracted and re-wrapped with a type assertion.
"""
function construct_and_define_function(signature::Expr, expressions::Vector{Expr}, return_expr::Expr, return_type::DataType)
    body = quote end
    for expr in expressions
        push!(body.args, expr)
    end
    ret_val = (return_expr.head == :return) ? return_expr.args[1] : return_expr
    typed_return = Expr(:return, Expr(:(::), ret_val, return_type))
    push!(body.args, typed_return)
    complete_function = Expr(:function, signature, body)
    @eval $complete_function
end

"""
Return a default zero-equivalent value for the given type.
"""
default_value(::Type{Bool}) = false
default_value(T::DataType) = T(0)


"""
Create a function expression wrapping generated body code in a typed,
callable function skeleton with initialized temps and outputs.

Returns an Expr that can be @eval'd to define the function.
"""
function create_harness(s::GenState, generated_body::Vector, func_name::Symbol; check_outputs::Bool=false)
	sorted_inputs = sort(collect(s.inputs), by=first)
	sorted_outputs = sort(collect(s.outputs), by=first)

	input_expr = [Expr(:(::), name, T) for (name, T) in sorted_inputs]

	body_expressions = Vector{Any}()

	for (v, T) in sorted_outputs
		push!(body_expressions, :($v = $(default_value(T))))
	end

	for (v, T) in s.temps
		push!(body_expressions, :($v = $(default_value(T))))
	end

	append!(body_expressions, generated_body)

	if check_outputs
		for (name, T) in sorted_outputs
			push!(body_expressions, Expr(:macrocall, Symbol("@assert"), nothing, Expr(:call, :isa, name, T)))
		end
	end

	return_vars = [name for (name, _) in sorted_outputs]
	if length(return_vars) == 1
		push!(body_expressions, Expr(:return, return_vars[1]))
	else
		push!(body_expressions, Expr(:return, Expr(:tuple, return_vars...)))
	end

	return Expr(:function, Expr(:call, func_name, input_expr...), Expr(:block, body_expressions...))
end


###################################################################################
# Loop safety: LoopLimitExceeded and add_loop_checks.
###################################################################################

"""
Custom exception thrown when a loop exceeds its iteration limit.
"""
struct LoopLimitExceeded <: Exception end

"""
    add_loop_checks_expr(expr, limit)

Recursively instrument an expression tree, wrapping each :for and :while node
with an iteration counter and a check that throws LoopLimitExceeded if the
counter exceeds `limit`.
"""
function add_loop_checks_expr(expr::Expr, limit::Int)
	if expr.head == :for || expr.head == :while
		counter = gensym("__loopcheck")
		increment = :($counter += 1)
		check = Expr(:if, Expr(:call, :>, counter, limit), Expr(:call, :throw, Expr(:call, :LoopLimitExceeded)))

		body_idx = 2
		processed_body = add_loop_checks_expr(expr.args[body_idx], limit)

		if processed_body isa Expr && processed_body.head == :block
			new_body = Expr(:block, increment, check, processed_body.args...)
		else
			new_body = Expr(:block, increment, check, processed_body)
		end

		if expr.head == :for
			new_loop = Expr(:for, expr.args[1], new_body)
		else
			processed_cond = add_loop_checks_expr(expr.args[1], limit)
			new_loop = Expr(:while, processed_cond, new_body)
		end

		return Expr(:block, :($counter = 0), new_loop)

	elseif expr.head == :block
		return Expr(:block, [add_loop_checks_expr(a, limit) for a in expr.args]...)

	else
		new_args = [a isa Expr ? add_loop_checks_expr(a, limit) : a for a in expr.args]
		return Expr(expr.head, new_args...)
	end
end

add_loop_checks_expr(x, limit::Int) = x

"""
    add_loop_checks(body; limit=10_000)

Instrument a vector of body expressions with loop iteration checks.
Returns a new vector (the original is not modified).
"""
function add_loop_checks(body; limit::Int=10_000)
	return [add_loop_checks_expr(expr, limit) for expr in deepcopy(body)]
end


###################################################################################
# Tree utilities.
###################################################################################

"""
    unravel(tree, expressions=[])

Flatten an Expr tree into a list of all sub-expressions via pre-order traversal.
"""
function unravel(tree, expressions=[])
	if tree isa Expr
		push!(expressions, tree)
		for arg in tree.args
			unravel(arg, expressions)
		end
	end
	return expressions
end
