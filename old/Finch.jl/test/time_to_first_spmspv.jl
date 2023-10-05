using Finch
y = @fiber d(e(0.0))
A = @fiber d(sl(e(0.0)))
x = @fiber sl(e(0.0))

#using ProfileView
#@profview @elapsed Finch.execute_code(:ex, typeof(Finch.@finch_program_instance @loop i j y[i] += A[i, j] * x[i]))

first_run = @elapsed Finch.execute_code(:ex, typeof(Finch.@finch_program_instance @loop i j y[i] += A[i, j] * x[i]))
second_run = @elapsed Finch.execute_code(:ex, typeof(Finch.@finch_program_instance @loop i j y[i] += A[i, j] * x[i]))

using MethodAnalysis
using RewriteTools

finch_methods = length(methodinstances(Finch))
rewrite_methods = length(methodinstances(RewriteTools))

@info "results" first_run second_run finch_methods rewrite_methods
# Starting point
#┌ Info: results
#│   first_run = 67.862103246
#│   second_run = 0.22789351
#│   tests = 7m22.4s
#│   finch_methods = 866
#└   rewrite_methods = 1788
# With precompilation
#┌ Info: results
#│   first_run = 43.688420017
#│   second_run = 0.137507335
#│   tests = 6m49.2s
#│   finch_methods = 866
#└   rewrite_methods = 1784
# After deparameterizing access and cleaning up:
#┌ Info: results
#│   first_run = 39.680086167
#│   second_run = 0.138437737
#│   tests = 4m04.4s
#│   finch_methods = 994
#└   rewrite_methods = 1612
#After enforcing Literals and Values:
#┌ Info: results
#│   first_run = 32.952320614
#│   second_run = 0.785031029
#│   tests = 3m46.5s
#│   finch_methods = 991
#└   rewrite_methods = 917
#After unityping Literals and Values:
#┌ Info: results
#│   first_run = 33.424417358
#│   second_run = 0.784262053
#│   tests = 3m35.3s
#│   finch_methods = 1011
#└   rewrite_methods = 716
#After unityping With
#┌ Info: results
#│   first_run = 33.423286476
#│   second_run = 0.767126124
#│   tests = 3m35.0s
#│   finch_methods = 953
#└   rewrite_methods = 722
#After unityping Access, Virtual, and Chunk
#┌ Info: results
#│   first_run = 30.872168248
#│   second_run = 0.790361734
#│   tests = 3m24.2s
#│   finch_methods = 998
#└   rewrite_methods = 569
#After unityping Loop, Sieve, and Assign
#┌ Info: results
#│   first_run = 26.72391913
#│   second_run = 0.765605838
#│   tests = 3m13.1s
#│   finch_methods = 961
#└   rewrite_methods = 531
#All Unityped
#┌ Info: results
#│   first_run = 21.337957409
#│   second_run = 0.765421906
#│   tests = 2m39.2s
#│   finch_methods = 804
#└   rewrite_methods = 318
#Remove IndexNode
#┌ Info: results
#│   first_run = 21.343022603
#│   second_run = 0.744349017
#│   tests = 2m31.4s
#│   finch_methods = 819
#└   rewrite_methods = 305
#Remove unresolve/resyntax
#┌ Info: results
#│   first_run = 21.224471498
#│   second_run = 0.067946879
#│   tests = 2m09.2s
#│   finch_methods = 820
#└   rewrite_methods = 305