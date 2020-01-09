import torch

from qpth.qp import QPFunction

import pdb

def get_fowards_function(soft_nonnegativity, soft_lambdas, l1_softness,
        force_max_dissipation_qp, force_max_dissipation_ext):

    def forward(params, data):
        mu = params.mu

        n = data.shape[0]
        vs = torch.unsqueeze(data[:, 0], 1)
        vnexts = torch.unsqueeze(data[:, 1], 1)
        us = torch.unsqueeze(data[:, 2], 1)

        beta = vnexts - vs - us
        
        pad_base = torch.zeros(6)

        G = torch.tensor([[1.0, -1, 1], [-1, 1, 1], [-1, -1, 0]]).repeat(n, 1, 1)
        
        Gpad = torch.tensor([[1.0, -1, 1, 0, 0, 0, 0, 0, 0],
                             [-1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [-1, -1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(n, 1, 1)
        
        fmats = torch.tensor([[1.0, 1, 0], 
                              [-1, -1, 0],
                              [0, 0, 1]]).unsqueeze(0).repeat(n, 1, 1)
        fvecs = torch.cat((vs, us, mu * torch.ones(vs.shape)), 1).unsqueeze(2)
        f = torch.bmm(fmats, fvecs)
        batch_zeros = torch.zeros(f.shape)
        fpad = torch.cat((f, batch_zeros, batch_zeros), 1)
        
        # For prediction error
        A = torch.tensor([[1.0, -1, 0, 0, 0, 0, 0, 0, 0],
                          [-1, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(n, 1, 1)
        beta_zeros = torch.zeros(beta.shape)
        b = torch.cat((-2 * beta, 2 * beta, beta_zeros, beta_zeros, beta_zeros, 
                       beta_zeros, beta_zeros, beta_zeros, beta_zeros), 1).unsqueeze(2)

        a1 = 1
        a2 = 1
        a3 = 20
        a4 = 20
        a5 = 0.1

        if l1_softness:
            inequality_slack_penalty = torch.tensor([0.0, 0, 0, 1, 1, 1, 0, 0, 0]).repeat(n, 1).unsqueeze(2)
            lambdas_slack_penalty = torch.tensor([0.0, 0, 0, 0, 0, 0, 1, 1, 1]).repeat(n, 1).unsqueeze(2)

            Q = 2 * a1 * A + 2 * a2 * Gpad  
            p = a1 * b + a2 * fpad + a3 * inequality_slack_penalty + a4 * lambdas_slack_penalty
        else:
            inequality_slack_penalty = torch.diag(torch.tensor([0.0, 0, 0, 1, 1, 1, 0, 0, 0])).repeat(n, 1, 1)
            lambdas_slack_penalty = torch.diag(torch.tensor([0.0, 0, 0, 0, 0, 0, 1, 1, 1])).repeat(n, 1, 1)

            Q = 2 * a1 * A + 2 * a2 * Gpad + a3 * inequality_slack_penalty + a4 * lambdas_slack_penalty 
            p = a1 * b + a2 * fpad

        
        # Constrain slacks (but not lambdas) to be >= 0
        R = torch.cat((torch.zeros(6, 3), -torch.eye(6)), 1).repeat(n, 1, 1)
        h = torch.zeros((1, 6)).repeat(n, 1).unsqueeze(2)

        I = torch.eye(3).repeat(n, 1, 1)

        if soft_nonnegativity:
            # Constrain G lambda + s(1:3) >= 0
            R = torch.cat((R, -torch.cat((G, I, torch.zeros(n, 3, 3)), 2)), 1)
        else:
            # Constrain G lambda + f >= 0
            R = torch.cat((R, torch.cat((-G, torch.zeros(n, 3, 6)), 2)), 1)

        # This is the same with soft or hard nonnegativity constraint
        h = torch.cat((h, f), 1)
        
        if soft_lambdas:
            # Constrain lambda + s(4:7) >= 0
            R = torch.cat((R, -torch.cat((I, torch.zeros(n, 3, 3), I), 2)), 1)
            h = torch.cat((h, torch.zeros(n, 3, 1)), 1)
        else:
            # Constrain lambda >= 0
            R = torch.cat((R, torch.cat((-I, torch.zeros(n, 3, 6)), 2)), 1)
            h = torch.cat((h, torch.zeros(n, 3, 1)), 1)
        
        if force_max_dissipation_qp:
            # Factor 2 for 1/2 in front of quadratic term
            max_dissipation_mat = a5 * 2 * torch.tensor([[1.0, 1, 0, 0, 0, 0, 0, 0, 0],
                                              [1, 1, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(n, 1, 1) 
            max_dissipation_vec = a5 * torch.tensor([[-2 * mu, -2 * mu, 0, 0, 0, 0, 0, 0, 0]]).repeat(n, 1).unsqueeze(2) 

            Q = Q + max_dissipation_mat
            p = p + max_dissipation_vec


        Qmod = 0.5 * (Q + Q.transpose(1, 2)) + 0.001 * torch.eye(9).repeat(n, 1, 1)
        
        z = QPFunction(check_Q_spd=False)(Qmod, p.squeeze(2), R, h.squeeze(2), torch.tensor([]), torch.tensor([]))

        lcp_slack = torch.bmm(Gpad, z.unsqueeze(2)) + fpad

        costs = 0.5 * torch.bmm(z.unsqueeze(1), torch.bmm(Qmod, z.unsqueeze(2))) \
                + torch.bmm(p.transpose(1, 2), z.unsqueeze(2)) + a1 * beta**2

        if force_max_dissipation_qp:
            max_dissipation_const_cost = a5 * mu ** 2
            max_dissipation_cost = 0.5 * torch.bmm(z.unsqueeze(1), torch.bmm(max_dissipation_mat, z.unsqueeze(2))) \
                + torch.bmm(max_dissipation_vec.transpose(1, 2), z.unsqueeze(2)) + max_dissipation_const_cost
            # max dissipation cost is already in QP (just outputted for debugging), need to add const term
            costs = costs + torch.ones(costs.shape) * max_dissipation_const_cost

        if force_max_dissipation_ext:
            lambda_plus = z[:, 0]
            lambda_minus = z[:, 1]
            mgmu = mu * torch.ones(lambda_plus.shape)

            max_dissipation_cost = a5 * (mgmu - lambda_plus - lambda_minus) ** 2
            costs = costs + max_dissipation_cost

        outputs = {'cost': sum(costs), 'lambdaPlus': z[0,0],
                'lambdaMinus': z[0, 1], 'gamma': z[0, 2],
                'slack[0]': lcp_slack[0,0], 'slack[1]': lcp_slack[0, 1],
                'slack[2]': lcp_slack[0, 2]}
        
        if force_max_dissipation_qp:
            outputs['max_diss_cost'] = max_dissipation_cost 
        if force_max_dissipation_ext:
            outputs['max_diss_cost'] = sum(max_dissipation_cost)

        return torch.sum(costs), outputs

    return forward
