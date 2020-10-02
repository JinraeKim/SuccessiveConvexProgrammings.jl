module SuccessiveConvexProgrammings


using Reexport
include("SuccessiveConvexification.jl")
@reexport using .SuccessiveConvexification
# Note: module Linearisers is reexported in Algorithms


end
