module SuccessiveConvexProgrammings


using Reexport
include("SuccessiveConvexifications.jl")
@reexport using .SuccessiveConvexifications
# Note: module Linearisers is reexported in Algorithms


end
