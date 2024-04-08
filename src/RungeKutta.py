import numpy as np

def RK_perso(param, t, y, Tmax, Pcore):
    dydt = np.zeros(5)
    Ta = Tmax * param.profil(t) + 273
    tau, w, R2w, r2w, T = y
    R = np.sqrt(R2w / w)
    r = np.sqrt(r2w / w)
    mu = viscosite(T)
    dwdz = tau / (3 * np.pi * mu * (R**2 - r**2))
    dydt[2] = (Pcore * r**2 * R**2 - param.gamma * r * R * (r + R)) / (mu * (R**2 - r**2))
    dydt[0] = param.rho * np.pi * (R**2 - r**2) * (w * dwdz - param.g) - param.gamma * np.pi * (dydt[2] - R**2 * dwdz) / (2 * R * w) - param.gamma * np.pi * (dydt[2] - r**2 * dwdz) / (2 * r * w)
    dydt[1] = dwdz
    dydt[3] = dydt[2]
    dydt[4] = (R * param.N * (Ta - T) + R * param.sigma * param.alpha * (Ta**4 - T**4)) * 2 / (R**2 - r**2) / (param.rho * param.Cp * w)
    return dydt
