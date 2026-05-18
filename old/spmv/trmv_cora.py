import numpy as np
import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
#sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import gc
import math
import numpy as np
np.random.seed(0)

def ceildiv(a, b):
    import tvm
    if isinstance(a, int) and isinstance(b, int):
        return (a + b - 1) // b
    else:
        return tvm.floordiv(a + b - 1, b)

def ceilmult(a, b):
    import tvm
    try:
        ai = int(a)
        bi = int(b)
        return bi * ((ai + bi - 1) // bi)
    except:
        return b * tvm.floordiv(a + b - 1, b)

def floormult(a, b):
    import tvm
    if isinstance(a, int) and isinstance(b, int):
        return b * (a // b)
    else:
        return b * tvm.floordiv(a, b)

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def stats(arr):
    # return np.min(arr), np.mean(arr), np.max(arr)
    # return np.min(arr), np.max(arr)
    return np.mean(arr)

def int_shape(expr_shape, rmap):
    import tvm
    from tvm.tir import ir_pass
    shape = []
    for e in expr_shape:
        e1 = e
        # print(1, e1, rmap)
        e2 = ir_pass.Substitute(e1, rmap)
        # print(2, e2)
        e3 = ir_pass.Simplify(e2)
        # print(3, e3)
        shape.append(int(e3))
    return shape

def get_shape(t, rmap):
    import tvm
    if isinstance(t, tuple) or isinstance(t, list):
        return t
    elif isinstance(t, tvm.te.Tensor):
        return int_shape(t.shape, rmap)
    elif isinstance(t, tvm.tir.Buffer):
        return int_shape(t.shape.dense_shape(), rmap)
    else:
        print(t)
        assert False

def create_ragged_array(dense_shape, flat_size, dtype, ctx):
    # print("YO1: ", flat_size)
    import tvm
    # src_np_array = np.random.default_rng().random((flat_size,), dtype=np.float32)
    src_np_array = np.random.normal(size=(flat_size,)).astype(dtype)
    # src_np_array = np.full((flat_size,), 0.1, dtype).astype(dtype)
    tvm_array = tvm.nd.ragged_empty(dense_shape, flat_size, dtype=dtype, ctx=ctx)
    tvm_array.copyfrom(src_np_array, is_dst_ragged=True)
    del src_np_array
    return tvm_array

def create_numpy_array(t, dtype, rmap={}, lw_args=None):
    shape = get_shape(t, rmap)
    # print("YO2: ", shape)
    # return np.zeros(shape, dtype)
    # return np.full(shape, 0.1, dtype)
    return np.random.normal(size=shape, loc=0, scale=4).astype(dtype)

def create_tvm_array(t, dtype, ctx, rmap={}, lw_args=None):
    import tvm
    shape = get_shape(t, rmap)

    assert (lw_args is not None)
    if t in lw_args:
        flat_size = lw_args[t]
        # print(t, flat_size, shape)
        return create_ragged_array(shape, flat_size, dtype, ctx)

    # return np.zeros(shape, dtype)
    # return tvm.nd.array(np.full(shape, 0.1, dtype), ctx)
    # print("YO3: ", shape)
    # np_array = np.random.default_rng().random(shape, dtype=np.float32)
    np_array = np.random.normal(size=shape).astype(dtype)
    # np_array = np.full(shape, 0.1, dtype)
    tvm_array = tvm.nd.array(np_array, ctx)
    del np_array
    return tvm_array
    # return tvm.nd.array(np.random.sample(size=shape), ctx)

def get_ctx(target):
    import tvm
    ctx = None
    if target.startswith('llvm') or target == 'c':
        ctx = tvm.cpu(0)
    elif target.startswith('cuda'):
        ctx = tvm.gpu(0)
    else:
        raise ValueError('Unsupported target %s' % target)
    return ctx

def mean(l):
    return sum(l) / len(l)

def execute(target, built, inputs, ctx, debug = False):
    if debug:
        if target == 'c':
            built['default_function'](*inputs)
        else:
            built(*inputs)
        ctx.sync()
        return -100000000
    else:
        if target == 'c':
            built['default_function'](*inputs)
            return -100000000
            evaluator = built.time_evaluator('default_function', ctx, 1, repeat=10)
        else:
            # evaluator = built.time_evaluator(built.entry_name, ctx, repeat=5, number=20)
            evaluator = built.time_evaluator(built.entry_name, ctx, repeat=5, number=100)
        eval_result = evaluator(*inputs)
        return min(list(eval_result.results)[1:])
        #return mean(list(eval_result.results)[1:]) * 1000
        # return mean(list(eval_result.results)) * 1000

def is_ragged(t):
    import tvm
    if isinstance(t, tvm.te.Tensor):
        if t.op.output_layout(0) is None:
            return False
        else:
            return t.op.output_layout(0).is_ragged()
    elif isinstance(t, tvm.tir.Buffer):
        return t.shape.is_ragged()
    else:
        print(t)
        assert False

def reverse_sort_batches(batches):
    ret = []
    for batch in batches:
        ret.append(np.sort(batch)[::-1].astype('int32'))
    return ret

def append_padded_sum(batches, factor):
    ret = []
    for batch in batches:
        batch_sum = np.sum(batch)
        padding_length = ceilmult(batch_sum, factor) - batch_sum
        if padding_length == 0: padding_length = factor
        padded = np.append(batch, padding_length).astype('int32')
        # print('PADDING', padding_length, batch_sum, np.sum(padded))
        ret.append(padded)
    return ret

def run_trmm(built, i_inputs_tensors, t_inputs_tensors, lw_args, args, pad_sum=None):
    import tvm
    ctx = get_ctx(args.target)
    cpu_ctx = get_ctx("llvm")

    host_i_inputs, dev_i_inputs = [], []
    if len(i_inputs_tensors) == 2:
        host_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), cpu_ctx) for i in i_inputs_tensors[0]]
        dev_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), ctx) for i in i_inputs_tensors[1]]

    t_inputs = [create_tvm_array(i, "float32", ctx, lw_args={}) for i in t_inputs_tensors]
    inputs = t_inputs + host_i_inputs + dev_i_inputs
    time = execute(args.target, built, inputs, ctx, args.debug)
    
    print("RESULTS", time, sep=',')

    with open("spmv_results_cora.json", "w") as f:
        f.write(
            """[
    {{
        "time": {},
        "method": "cora",
        "kernel": "spmv",
        "matrix": "upper_triangle",
        "dataset": "triangle"
    }}
]""".format(time)
        )

    for i in range(len(t_inputs)):
        size_fn = {}
        target = None
        if t_inputs_tensors[i] in size_fn:
            target = np.empty(size_fn[t_inputs_tensors[i]], dtype='float32')
        t_inputs[i] = t_inputs[i].asnumpy(target=target, is_src_ragged=is_ragged(t_inputs_tensors[i]))
    return t_inputs

def lower_or_build(name, s, inputs, args, prep_code_mode='with_prep_code', binds=None,
                   size_fn={}, pad_sum=None, substitutes=None, run_function=run_trmm, hoist_loads=False):
    import tvm
    prep_code_mode = 'only_prep_code' if args.only_prep_code else prep_code_mode
    with tvm.build_config(prep_code_mode=prep_code_mode,
                          fill_in_function_bodies=not args.debug_functions,
                          hoist_loads=hoist_loads,
                          disable_assert=args.disable_assert if hasattr(args, 'disable_assert') else False):
        if args.debug_code == 'ir':
            lowered = tvm.lower(s, inputs, args.target, simple_mode=True, binds=binds, substitutes=substitutes)
            print(lowered)
            return None, None
        elif args.debug_code == 'code':
            fadd, _ = tvm.build(s, inputs, args.target, binds=binds)
            if args.target == 'cuda':
                print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
            else:
                print('-----CPU code-----\n' + fadd.get_source())
            return None, None
        else:
            assert args.debug_code is None
            fadd, i_bufs = tvm.build(s, inputs, args.target, binds=binds, substitutes=substitutes)
            # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/attn_v.so')
            return run_function(fadd, i_bufs, inputs[1], size_fn, args, pad_sum=pad_sum)



parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--m', dest='m', default=1024, type=int)
parser.add_argument('--n', dest='n', default=1, type=int)
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--load-balance', dest='load_balance', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--optimize', dest='optimize', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--only-prep-code', dest='only_prep_code', default=False, action='store_true')
parser.add_argument('--m1', dest='m1', default=2, type=int)
parser.add_argument('--m2', dest='m2', default=1, type=int)
parser.add_argument('--n1', dest='n1', default=32, type=int)
parser.add_argument('--n2', dest='n2', default=4, type=int)
parser.add_argument('--k1', dest='k1', default=8, type=int)
parser.add_argument('--k2', dest='k2', default=8, type=int)
args = parser.parse_args()

M = args.m
N = args.n
md = Dim('md')
nd = Dim('nd')
kd = Dim('kd')

ls =  {
    0: Uf.from_constant('md', M, 'l'),
    1: Uf.from_constant('nd', N, 'l'),
    2: Uf.from_constant('kd', M, 'l'),
}

loop_ufs=[ls[0], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((M, M), [md, kd], loop_ufs, name='A', width_ufs=None)

B = te.placeholder((M, N), name='B')

alpha = 2
def len_ufw(name, pad): return Ufw(name, "l", (pad, M), [md], [], lambda: lambda m: ceilmult(m + 1, pad))
luf = len_ufw('s2k', 256).get_uf()

loop_ufs=[ls[0], ls[1]]
S = te.ragged_compute((M, N), [md, nd], loop_ufs,
                      lambda ds, rds: tvm.sum(tvm.tir.Cast('int32', rds['k'] < (ds[md] + 1)) *
                                              A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                              axis=rds['k'], dimensions = [kd]),
                      # lambda ds, rds: tvm.sum(A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                              # axis=rds['k'], dimensions = [kd]),
                      name = 'S', reduce_axis_ufs = [('k', luf)], width_uf_lists=None)

O = te.ragged_compute((M, N), [md, nd], loop_ufs, lambda ds: alpha*S[ds[md], ds[nd]], name = 'O', width_uf_lists=None)

s = tvm.create_schedule([O.op])

def schedule_op(O, suffix, cache_write_tensor=None):
    if cache_write_tensor is not None:
        O_local = cache_write_tensor
    else:
        O_local, = s.cache_write([O], "local", storage_layout_mode='loop_layout')

    O_local_m_c, O_local_n_c, O_local_k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
    O_local_m_c_o_i, O_local_m_c_i = s[O_local].split(O_local_m_c, factor=4)
    O_local_m_c_o_o_i, O_local_m_c_o_i = s[O_local].split(O_local_m_c_o_i, factor=64)
    O_local_m_c_o_o_o, O_local_m_c_o_o_i = s[O_local].split(O_local_m_c_o_o_i, factor=1)

    O_local_n_c_o_i, O_local_n_c_i = s[O_local].split(O_local_n_c, factor=64)
    O_local_n_c_o_o_i, O_local_n_c_o_i = s[O_local].split(O_local_n_c_o_i, factor=1)
    O_local_n_c_o_o_o, O_local_n_c_o_o_i = s[O_local].split(O_local_n_c_o_o_i, factor=1)

    O_local_k_o, O_local_k_i = s[O_local].split(O_local_k, factor=32)
    s[O_local].reorder(O_local_m_c_o_o_o, O_local_n_c_o_o_o, O_local_m_c_o_o_i, O_local_n_c_o_o_i, O_local_k_o, O_local_m_c_o_i, O_local_n_c_o_i, O_local_k_i, O_local_m_c_i, O_local_n_c_i)

    O_m, O_n, = tuple(O.op.axis)
    O_m_o, O_m_i = s[O].split(O_m, factor=256)
    O_n_o, O_n_i = s[O].split(O_n, factor=64)
    s[O].reorder(O_m_o, O_n_o, O_m_i, O_n_i)

    O_m_o_n_o_fused = s[O].fuse(O_m_o, O_n_o)
    s[O].parallel(O_m_o_n_o_fused)
    s[O_local].compute_at(s[O], O_m_o_n_o_fused)
    s[O_local].pragma(O_local_m_c_o_o_o, "auto_unroll_max_step", 512)
    s[O_local].pragma(O_local_m_c_o_o_o, "unroll_explicit", True)
    s[O_local].vectorize(O_local_n_c_i)
    if cache_write_tensor is None: return [O.op, O_local.op]
    else: return []

substitute_ops = []
if args.optimize:
    substitute_ops += schedule_op(O, '', S)

inputs = [[], [A, B, O]]

substitutes=None
if args.load_balance:
    print('Load balancing')
    max_by = (M // 256) * (M // 64)
    substitutes=[substitute_ops, {'iO1_0.o1.o_f': Uf('sub', "", (0, max_by), [Dim('dum')], lambda b: max_by - b - 1)}]

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = lower_or_build(name, s, inputs, args, run_function=run_trmm,
                               prep_code_mode='no_prep_code', substitutes=substitutes)

# if args.op_split:
#     A, B, O1, O2  = out
#     for i in range(args.m):
#         print(i + 1, np.mean(O1[i, 0:(i+1)]), np.mean(O2[i, 0:(i+1)]))
# else:
#     A, B, O  = out
#     for i in range(args.m):
#         print(i + 1, np.mean(O[i, 0:(i+1)]))
