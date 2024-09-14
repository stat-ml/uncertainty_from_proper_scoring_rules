import numpy as np


def total_risk(g, grad_g, prob_gt, prob_pred):
    prob_gt = np.expand_dims(prob_gt, axis=1)
    prob_pred = np.expand_dims(prob_pred, axis=0)
    G_grad = grad_g(prob_pred)
    G = g(prob_pred)
    R_tot = np.sum(G_grad * (prob_pred - prob_gt), axis=-1, keepdims=True) - G
    return R_tot


def excess_risk(g, grad_g, prob_gt, prob_pred):
    prob_gt = np.expand_dims(prob_gt, axis=1)
    prob_pred = np.expand_dims(prob_pred, axis=0)
    G_grad = grad_g(prob_pred)
    G = g(prob_pred)
    G_tr = g(prob_gt)
    R_exc = G_tr - G + np.sum(G_grad * (prob_pred - prob_gt), axis=-1, keepdims=True)
    return R_exc


def bayes_risk(g, prob_gt):
    R_bayes = -g(prob_gt)
    return R_bayes
