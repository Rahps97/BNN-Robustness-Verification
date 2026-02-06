import qubovert as qv
import numpy as np
from qubovert import boolean_var
from utils import to_boolean
from qubovert._pcbo import _special_constraints_le_zero, _get_bounds, num_bits
from qubovert.utils._warn import QUBOVertWarning
from math import log2

def binarize(inp):
    output = inp.new(inp.size())
    output[inp >= 0] = 1
    output[inp < 0] = -1

    return output


# FIXME: Implement the following for quso
def le_zero_constraint_to_eq_zero_constraint(P, log_trick=True, bounds=None,
                                             suppress_warnings=False):
    # Inspired by PCBO.add_constraint_le_zero implementation.
    model = qv.PCBO()
    P = qv.PUBO(P)

    bounds = min_val, max_val = _get_bounds(P, bounds)
    if _special_constraints_le_zero(model, P, 1, log_trick, bounds):
        return model

    if min_val > 0:
        if not suppress_warnings:
            QUBOVertWarning.warn("Constraint cannot be satisfied")
        model += P
    elif max_val <= 0:
        if not suppress_warnings:
            QUBOVertWarning.warn("Constraint is always satisfied")
    else:
        # don't mutate the P that we put in model._constraints
        P = P.copy()
        if min_val:
            for i in range(num_bits(-min_val, log_trick)):
                v = pow(2, i) if log_trick else 1
                P[(model._next_ancilla,)] += v
                max_val += v
        model += P
        return model

    return model


def add_sign_constraint(H, count, partial_poly, output_bool, lam, k_layer, j):
    # FIXME check if can be simplified since some vars are constant
    if count == 3:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_2 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2, lam=lam)
    elif count == 7:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_4 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4,
                                 lam=lam)
    elif count == 15:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_8 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8, lam=lam)
    elif count == 31:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_16 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16, lam=lam)
    elif count == 63:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_32 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32,
                                 lam=lam)
    elif count == 127:
        aux = 1
        aux_1 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 32
        aux_32 = boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_64 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32
                                 - 64 * aux_64, lam=lam)
    elif count == 255:
        aux = 1
        aux_1 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 32
        aux_32 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 64
        aux_64 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_128 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32
                                 - 64 * aux_64 - 128 * aux_128, lam=lam)
    elif count == 511:
        aux = 1
        aux_1 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 32
        aux_32 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 64
        aux_64 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 128
        aux_128 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_256 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32
                                 - 64 * aux_64 - 128 * aux_128 - 256 * aux_256,
                                 lam=lam)
    elif count == 1023:
        aux = 1
        aux_1 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 2
        aux_2 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 4
        aux_4 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 8
        aux_8 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 16
        aux_16 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 32
        aux_32 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 64
        aux_64 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 128
        aux_128 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux = 256
        aux_256 = qv.boolean_var(f'aux_matrix_product_{k_layer}_{j}_{aux}')
        aux_512 = output_bool
        H.add_constraint_eq_zero(partial_poly - aux_1 - 2 * aux_2 - 4 * aux_4
                                 - 8 * aux_8 - 16 * aux_16 - 32 * aux_32
                                 - 64 * aux_64 - 128 * aux_128 - 256 * aux_256
                                 - 512 * aux_512,  lam=lam)
    else:
        raise NotImplementedError(f'count: {count}')

def add_argmax_sign_constraint(H, count, partial_poly, output_bool, lam, k_layer, label):
    # print("Values", np.log2(count))
    Sum = 0
    for power in range(int(np.log2(count)-1)):
        aux = 2 ** power
        Sum += aux * qv.boolean_var(f'aux_sum_{k_layer}_{label}_{aux}')
    aux = -(2 ** int(np.log2(count)-1))
    Sum += aux * output_bool
    # print("Sum", Sum)
    H.add_constraint_eq_zero(partial_poly - Sum, lam=lam)



def setup_optim_model(sample_input_spin, sample_input_target, model, args):
    """
    Setup the optimization model for the given neural network model and input data.
    
    Args:
        sample_input_spin (Tensor): The input data to the model.
        sample_input_target (Tensor): The target output for the model.
        model (nn.Module): The neural network model.
        args (Namespace): Command line arguments containing various parameters.

    Returns:
        H: The Hamiltonian representing the optimization problem.
        ordered_variables: A dictionary mapping variable indices to their names.
    """
    LAMBDA = args.LAMBDA
    epsilon = args.epsilon
    sample_input_boolean = to_boolean(sample_input_spin)
    qubo_vars = {}  # dict of variables
    H = qv.PCBO()

    gt = int(sample_input_target)

    sum_taus = 0
    for i in args.pixels_to_perturb:
        qubo_vars[f'tau_{i}'] = qv.boolean_var(f'tau_{i}')
        sum_taus += qubo_vars[f'tau_{i}']

    if args.objective == 'output':
        raise NotImplementedError('output')

    if args.include_perturbation_bound_constraint:
        H.add_constraint_lt_zero(sum_taus - epsilon, LAMBDA['perturbation_bound_constraint'])

    fc_in = sample_input_boolean
    for idx, (name, param) in enumerate(model.named_parameters()):
        # print(fc_in)
        param = param.data
        
        # loop over output dimension of layer
        for j in range(param.size()[0]):
            sum_partials = 0
            count_partials = 0
            # loop over input dimension of layer
            for i in range(param.size()[1]):
                weight = to_boolean(binarize(param[j][i]))
                qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = qv.boolean_var(f'partial_matrix_product_{idx}_{i}_{j}')
                # print(f'partial_matrix_product_{idx}_{i}_{j}')

                # First Layer
                if idx == 0:
                    # Adding the Perturbation
                    if i not in args.pixels_to_perturb:  
                        # tau_i is 0 i.e. no perturbation
                        if fc_in[i] == 1:
                            if weight == 0:
                                qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = 0

                            elif weight == 1:
                                qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = 1

                        elif fc_in[i] == 0:
                            if weight == 0:
                                qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = 1
                            elif weight == 1:
                                qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = 0

                    else:
                        # non-zero tau_i 
                        # TODO: if i not in range(3, 6) then assume tau_i is zero and subsequent vars
                        if fc_in[i] == 1:
                            if weight == 0:
                                H.add_constraint_eq_BUFFER(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], qubo_vars[f'tau_{i}'], lam=LAMBDA['hard_constraints'])
                            
                            elif weight == 1:
                                H.add_constraint_eq_NOT(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], qubo_vars[f'tau_{i}'], lam=LAMBDA['hard_constraints'])

                        elif fc_in[i] == 0:
                            if weight == 0:
                                H.add_constraint_eq_NOT(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], qubo_vars[f'tau_{i}'], lam=LAMBDA['hard_constraints'])

                            elif weight == 1:
                                H.add_constraint_eq_BUFFER(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], qubo_vars[f'tau_{i}'], lam=LAMBDA['hard_constraints'])
                # Subsequent Layers
                else:
                    if weight == 0:
                        H.add_constraint_eq_NOT(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], fc_in[i], lam=LAMBDA['hard_constraints'])

                    elif weight == 1:
                        H.add_constraint_eq_BUFFER(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], fc_in[i], lam=LAMBDA['hard_constraints'])

                sum_partials += qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}']
                count_partials += 1
            # All layers but Argmax
            if idx < len(list(model.modules())) - 2:
                # print(idx)
                qubo_vars[f'matrix_product_{idx}_{j}'] = qv.boolean_var(f'matrix_product_{idx}_{j}')
                output_bool = qubo_vars[f'matrix_product_{idx}_{j}']
                add_sign_constraint(H, count_partials, sum_partials, output_bool=output_bool, lam=LAMBDA['hard_constraints'], k_layer=idx, j=j)
            else:
                pass
        
        #Input for next Layer
        if idx < len(list(model.modules())) - 2:
            fc_in = [qubo_vars[f'matrix_product_{idx}_{j}'] for j in range(param.size()[0])]            
        
    # Argmax Layer Implementation
    # print(idx)
    # print(param.size())
    # print(gt)
    res = 0
    for k in range(param.size()[0]):
        sum_class = 0
        qubo_vars[f'argmax_sign_{idx}_{k}'] = qv.boolean_var(f'argmax_sign_{idx}_{k}')
        output_bool = qubo_vars[f'argmax_sign_{idx}_{k}']
        if k != gt:
            for l in range(param.size()[1]):
                sum_class += qubo_vars[f'partial_matrix_product_{idx}_{l}_{gt}']
                sum_class -= qubo_vars[f'partial_matrix_product_{idx}_{l}_{k}']
            # print("Count", 2*(param.size()[1]+1))
            add_argmax_sign_constraint(H, 2*(param.size()[1]+1), sum_class, output_bool, LAMBDA['hard_constraints'], idx, k)
            res += qubo_vars[f'argmax_sign_{idx}_{k}']
    H.add_constraint_gt_zero(res, lam=LAMBDA['hard_constraints'])

    ordered_variables = list(H.convert_solution([0]*len(H.to_qubo().variables)).keys())

    return H, ordered_variables


# def setup_optim_model(sample_input_spin, sample_input_target, model, args):

#     # Rahul: ONLY ZERO objective is tested
#     LAMBDA = args.LAMBDA
#     epsilon = args.epsilon
#     sample_input_boolean = to_boolean(sample_input_spin)
#     qubo_vars = {}  # dict of variables
#     H = qv.PCBO()

#     gt = int(sample_input_target)

#     sum_taus = 0
#     for i in args.pixels_to_perturb:
#         qubo_vars[f'tau_{i}'] = qv.boolean_var(f'tau_{i}')
#         sum_taus += qubo_vars[f'tau_{i}']

#     if args.objective == 'output':
#         raise NotImplementedError('output')

#     if args.include_perturbation_bound_constraint:
#         H.add_constraint_lt_zero(sum_taus - epsilon, LAMBDA['perturbation_bound_constraint'])

#     fc_in = sample_input_boolean
#     for idx, layer in enumerate(model.named_parameters()):
#         param = layer[1]
#         if idx == len(list(model.modules()))-1:
#             break
#         # loop over output dimension of layer
#         for j in range(param.size()[0]):
#             sum_partials = 0
#             count_partials = 0
#             # loop over input dimension of layer
#             for i in range(param.size()[1]):
#                 weight = to_boolean(binarize(param[j][i]))
#                 qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = qv.boolean_var(f'partial_matrix_product_{idx}_{i}_{j}')

#                 if idx == 0:
#                     if i not in args.pixels_to_perturb:  # tau_i is 0
#                         if fc_in[i] == 1:
#                             if weight == 0:
#                                 qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = 0

#                             elif weight == 1:
#                                 qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = 1

#                         elif fc_in[i] == 0:
#                             if weight == 0:
#                                 qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = 1
#                             elif weight == 1:
#                                 qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'] = 0

#                     else:
#                         # the first layer does not require xnor
#                         # TODO: if i not in range(3, 6) then assume tau_i is
#                         # zero and subsequent vars
#                         if fc_in[i] == 1:
#                             if weight == 0:
#                                 H.add_constraint_eq_BUFFER(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], qubo_vars[f'tau_{i}'], lam=LAMBDA['hard_constraints'])
                            
#                             elif weight == 1:
#                                 H.add_constraint_eq_NOT(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], qubo_vars[f'tau_{i}'], lam=LAMBDA['hard_constraints'])

#                         elif fc_in[i] == 0:
#                             if weight == 0:
#                                 H.add_constraint_eq_NOT(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], qubo_vars[f'tau_{i}'], lam=LAMBDA['hard_constraints'])

#                             elif weight == 1:
#                                 H.add_constraint_eq_BUFFER(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], qubo_vars[f'tau_{i}'], lam=LAMBDA['hard_constraints'])

#                 else:
#                     if weight == 0:
#                         H.add_constraint_eq_NOT(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], fc_in[i], lam=LAMBDA['hard_constraints'])

#                     elif weight == 1:
#                         H.add_constraint_eq_BUFFER(qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}'], fc_in[i], lam=LAMBDA['hard_constraints'])

#                 sum_partials += qubo_vars[f'partial_matrix_product_{idx}_{i}_{j}']
#                 count_partials += 1

#             if idx == len(list(model.modules()))-2:
#                 #lld : Last Layer Dimension(2^n)
#                 lld = int(log2((param.size()[1]+1)/2))+1
#                 P = 0
#                 for i in range(lld):
#                     qubo_vars[
#                         f'sum_matrix_product_{idx}_{j}_{2**i}'
#                         ] = qv.boolean_var(
#                             f'sum_matrix_product_{idx}_{j}_{2**i}')
#                     P += 2**i * qubo_vars[
#                         f'sum_matrix_product_{idx}_{j}_{2**i}']
#                 H.add_constraint_eq_zero(sum_partials - P, lam=LAMBDA[
#                     'hard_constraints'])
#             else:
#                 qubo_vars[
#                 f'matrix_product_{idx}_{j}'
#                 ] = qv.boolean_var(f'matrix_product_{idx}_{j}')
#                 output_bool = qubo_vars[f'matrix_product_{idx}_{j}']
#                 add_sign_constraint(H, count_partials,
#                                     sum_partials,
#                                     output_bool=output_bool,
#                                     lam=LAMBDA['hard_constraints'],
#                                     k_layer=idx,
#                                     j=j)
#         if idx < len(list(model.modules()))-2:
#             fc_in = [qubo_vars[f'matrix_product_{idx}_{j}'
#                            ] for j in range(param.size()[0])]

#     # Argmax Layer Implementation
#     idx = len(list(model.modules())) - 2
#     residual = 0
#     for k in range(10):
#         if k != gt:
#             qubo_vars[
#                 f'argmax_sign_{idx}_{k}'
#                 ] = qv.boolean_var(f'argmax_sign_{idx}_{k}')
#             P = 0
#             for i in range(lld):
#                 qubo_vars[
#                     f'argmax_sign_{idx}_{k}_{2**i}'
#                     ] = qv.boolean_var(f'argmax_sign_{idx}_{k}_{2**i}')
#                 P += 2**i * (qubo_vars[f'sum_matrix_product_{idx}_{k}_{2**i}']
#                              - qubo_vars[f'sum_matrix_product_{idx}_{gt}_{2**i}'])
#             # Argmax Sum Variables
#             variable_sum = -8 * qubo_vars[f'argmax_sign_{idx}_{k}']
#             for i in range(lld):
#                 qubo_vars[
#                     f'argmax_sum_{idx}_{k}_{2**i}'
#                     ] = qv.boolean_var(f'argmax_sum_{idx}_{k}_{2**i}')
#                 variable_sum += 2**i * qv.boolean_var(
#                     f'argmax_sum_{idx}_{k}_{2**i}')
#             H.add_constraint_eq_zero(variable_sum - P,
#                                      lam=LAMBDA['hard_constraints'])
#             residual += qubo_vars[f'argmax_sign_{idx}_{k}']
#     H.add_constraint_gt_zero(9 - residual, lam=LAMBDA['hard_constraints'])

#     ordered_variables = list(H.convert_solution([0]*len(H.to_qubo().variables)).keys())

#     return H, ordered_variables
