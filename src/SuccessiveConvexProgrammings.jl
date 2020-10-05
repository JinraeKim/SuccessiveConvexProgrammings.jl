module SuccessiveConvexProgrammings


using Reexport
include("SuccessiveConvexifications.jl")
@reexport using .SuccessiveConvexifications
# Note: module Linearisers is reexported in Algorithms
include("Discretisers.jl")
@reexport using .Discretisers


end
