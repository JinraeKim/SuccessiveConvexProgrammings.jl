module Linearisers
"""
module Linearisers

To get jacobian, linearise dynamics, etc.
"""

using ForwardDiff

export get_jacobian


function get_jacobian(func, args...)
    jacob = zeros(length(func(args...)), 0)  # dummy calculation
    args_dim = 0
    for (i, arg) in enumerate(args)
        args_tmp = [args...]
        args_tmp[1], args_tmp[i] = args_tmp[i], args_tmp[1]
        altered_func = function(args2...)
            args_tmp2 = [args2...]
            args_tmp2[1], args_tmp2[i] = args_tmp2[i], args_tmp2[1]
            return func(args_tmp2...)
        end
        new_func = function(arg)
            return altered_func(arg, args_tmp[2:end]...)
        end
        jacob = cat(jacob, ForwardDiff.jacobian(new_func, arg), dims=2)
    end
    return jacob
end


end
