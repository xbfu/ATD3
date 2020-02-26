import numpy as np


def getData(table):
    """
    table -> data
    """
    # table = [s_f, v_f, s_l, v_l]
    # 0 - s_f: y of the FV at time t
    # 1 - v_f: velocity of the FV at time t
    # 2 - s_l: y of the LV at time t
    # 3 - v_l: velocity of the LV at time t
    table = np.array(table, dtype=np.float32)
    table_len = len(table)

    samples = []
    data = []

    for l in range(table_len - 2):
        v_f = table[l, 1]                                                       # v_f(t)
        v_f_obs = table[l + 1, 1]                                               # v_f(t + 1)
        delta_v = table[l, 3] - table[l, 1]                                     # Δv(t) = v_l(t) - v_f(t)
        delta_s = table[l, 2] - table[l, 0]                                     # Δs(t) = s_l(t) - s_f(t)
        v_l_next = table[l + 1, 3]                                              # v_l(t + 1)
        s_f = table[l, 0]                                                       # s_f(t)
        s_f_obs = table[l + 1, 0]                                               # s_f(t + 1)

        # sample = [v_f(t), Δv(t), Δs(t), v_l(t + 1), s_f(t), v_f(t + 1), s_f(t + 1)]
        sample = np.array([v_f, delta_v, delta_s, v_l_next, s_f, v_f_obs, s_f_obs])
        samples.append(sample)

    samples = np.array(samples, dtype=np.float32)
    samples = np.reshape(samples, newshape=[-1, 7])

    for i in range(table_len - 20):
        state = np.reshape(samples[i: i + 10, 0: 3], newshape=[1, -1])
        item = np.concatenate([state.squeeze(), samples[i + 9, 3: 7]])
        data.append(item)

    data = np.array(data, dtype=np.float32)

    return data